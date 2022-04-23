import copy
import math
import os
from abc import ABC
from typing import Any, Union, Tuple, List, Dict

import gym.spaces
import numpy as np
from rlgym.envs import Match
from rlgym.gym import Gym
from rlgym.utils import RewardFunction, ObsBuilder, StateSetter, TerminalCondition
from rlgym.utils.action_parsers import ActionParser, ContinuousAction, DefaultAction
from rlgym.utils.common_values import BLUE_TEAM
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.state_setters import StateWrapper, DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition

from rocket_learn.utils.util import encode_gamestate


class Agent(ABC):
    def step(self, car_id: int, gamestate: GameState):
        raise NotImplementedError

    def finish(self, result, is_evaluation=False):
        raise NotImplementedError


class RocketEnvObs(ObsBuilder):
    def __init__(self, team_sizes: list):
        super().__init__()
        self.team_sizes = team_sizes
        self.current_state = None
        self.current_enc_state = None

    def reset(self, initial_state: GameState):
        self.current_state = None
        self.current_enc_state = None

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        if state != self.current_state:
            enc_state = encode_gamestate(state)
            psi = 3 + GameState.BOOST_PADS_LENGTH + GameState.BALL_STATE_LENGTH
            pil = GameState.PLAYER_INFO_LENGTH
            b, o = self.team_sizes
            info = slice(0, psi)
            blue_team = slice(psi, psi + pil * b)
            orange_team = slice(psi + pil * 3, psi + pil * 3 + pil * o)
            enc_state = enc_state[np.r_[info, blue_team, orange_team]]
            self.current_enc_state = enc_state

        return player.car_id, self.current_enc_state


class RocketEnvSetter(StateSetter):
    def __init__(self, team_sizes: list, setter: StateSetter):
        self.team_sizes = team_sizes
        self.setter = setter

    def reset(self, state_wrapper: StateWrapper):
        b, o = self.team_sizes
        reduced_wrapper = StateWrapper(3, 3)

        # Now pass original objects so values change in original wrapper
        reduced_wrapper.ball = state_wrapper.ball
        reduced_wrapper.cars = state_wrapper.cars[:b] + state_wrapper.cars[3:3 + o]

        for car in state_wrapper.cars[b:3] + state_wrapper.cars[3 + 0:]:
            if car.team_num == BLUE_TEAM:
                car.set_pos((-1) ** b * b * 100, -6200, 100)
                car.set_rot(0, math.pi / 2, 0)
                b += 1
            else:
                car.set_pos((-1) ** b * b * 100, 6200, 100)
                car.set_rot(0, -math.pi / 2, 0)
                o += 1
            car.set_lin_vel(0, 0, 0)
            car.set_ang_vel(0, 0, 0)

        return state_wrapper


class RocketEnvAction(ContinuousAction):
    def __init__(self, team_sizes: list):
        super().__init__()
        self.team_sizes = team_sizes

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        assert len(actions) == sum(self.team_sizes)
        new_actions = np.zeros((6, 8))
        b = 0
        for i, act in enumerate(new_actions):
            if b <= self.team_sizes[0]:
                new_actions[i, :] = act
                b += 1
            else:
                new_actions[3 + i, :] = act
        return super(RocketEnvAction, self).parse_actions(actions, state)


def get_reduced_state(b, o, state):
    reduced_state = copy.deepcopy(state)
    reduced_state.players = (reduced_state.players[:b]  # Blue
                             + reduced_state.players[3:3 + o])  # Orange
    return reduced_state


class FilteredRewTermObsAct(RewardFunction, TerminalCondition, ObsBuilder, ActionParser, StateSetter):
    def __init__(
            self,
            team_sizes: list,
            reward_fn: RewardFunction,
            terminal_conditions: List[TerminalCondition],
            obs_builder: ObsBuilder,
            action_parser: ActionParser,
            state_setter: StateSetter
    ):
        super().__init__()
        self.team_sizes = team_sizes
        self.reward_fn = reward_fn
        self.terminal_conditions = terminal_conditions
        self.obs_builder = obs_builder
        self.action_parser = action_parser
        self.state_setter = state_setter

        self.current_state = None
        self.reduced_state = None

    def _update_state(self, state: GameState):
        if state != self.current_state:
            self.current_state = state
            self.reduced_state = get_reduced_state(*self.team_sizes, state)

    def reset(self, initial_state_or_wrapper: Union[GameState, StateWrapper]):
        # Reset in StateSetter has a different signature to the others
        if isinstance(initial_state_or_wrapper, StateWrapper):
            state_wrapper = initial_state_or_wrapper
            b, o = self.team_sizes
            reduced_wrapper = StateWrapper(3, 3)

            # Now pass original objects so values change in original wrapper
            reduced_wrapper.ball = state_wrapper.ball
            reduced_wrapper.cars = state_wrapper.cars[:b] + state_wrapper.cars[3:3 + o]
            self.state_setter.reset(reduced_wrapper)

            bc = oc = 0
            for car in state_wrapper.cars[b:3] + state_wrapper.cars[3 + o:]:
                if car.team_num == BLUE_TEAM:
                    bc += 1
                    car.set_pos((-1) ** bc * (bc // 2) * 100, -6200, 100)
                    car.set_rot(0, math.pi / 2, 0)
                else:
                    oc += 1
                    car.set_pos((-1) ** oc * (oc // 2) * 100, 6200, 100)
                    car.set_rot(0, -math.pi / 2, 0)
                car.set_lin_vel(0, 0, 0)
                car.set_ang_vel(0, 0, 0)
        elif initial_state_or_wrapper != self.current_state:
            self._update_state(initial_state_or_wrapper)
            assert self.reduced_state is not None
            self.reward_fn.reset(self.reduced_state)
            for tc in self.terminal_conditions:
                tc.reset(self.reduced_state)
            self.obs_builder.reset(self.reduced_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self._update_state(state)
        if player.car_id not in (p.car_id for p in self.reduced_state.players):
            return float("nan")
        return self.reward_fn.get_reward(player, self.reduced_state, previous_action)

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        self._update_state(state)
        if player.car_id not in (p.car_id for p in self.reduced_state.players):
            return None
        return self.obs_builder.build_obs(player, self.reduced_state, previous_action)

    def is_terminal(self, current_state: GameState) -> bool:
        self._update_state(current_state)
        return any(tc.is_terminal(self.reduced_state) for tc in self.terminal_conditions)

    def get_action_space(self) -> gym.spaces.Space:
        return self.action_parser.get_action_space()

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        assert len(actions) == sum(self.team_sizes)
        actions = self.action_parser.parse_actions(actions, state)
        new_actions = np.zeros((6, 8))
        b = o = 0
        for act in actions:
            if b < self.team_sizes[0]:
                new_actions[b, :] = act
                b += 1
            else:
                new_actions[3 + o, :] = act
                o += 1
        return new_actions


class FilteredGym(Gym):
    """
    The FilteredGym creates a 3v3 instance and sets gamemode by spawning them outside the field.
    This adds the ability to change gamemode within the same instance,
    as well as unfair matchups which are normally impossible (1v2, 1v3, 2v3, 0v2, 0v3)
    """

    def __init__(self, blue_players: int, orange_players: int, **match_kwargs):
        self._team_sizes = [blue_players, orange_players]
        self._filter = FilteredRewTermObsAct(
            self._team_sizes,
            match_kwargs.pop("reward_function", DefaultReward()),
            match_kwargs.pop("terminal_conditions", [TimeoutCondition(225), GoalScoredCondition()]),
            match_kwargs.pop("obs_builder", DefaultObs()),
            match_kwargs.pop("action_parser", DefaultAction()),
            match_kwargs.pop("state_setter", DefaultState())
        )
        match = Match(
            reward_function=self._filter,
            terminal_conditions=self._filter,
            obs_builder=self._filter,
            action_parser=self._filter,
            state_setter=self._filter,
            team_size=3,
            self_play=True,
            **match_kwargs
        )
        super().__init__(match, os.getpid(), use_injector=True)

    def reset(self, return_info=False, blue_players=None, orange_players=None) -> Union[List, Tuple]:
        if blue_players is not None:
            self._team_sizes[0] = blue_players
        if orange_players is not None:
            self._team_sizes[1] = orange_players
        if return_info:
            obs, info = super(FilteredGym, self).reset(True)
            state = info.get("state")
            if state is not None:
                info["state"] = get_reduced_state(*self._team_sizes, state)
            return obs, info
        return super(FilteredGym, self).reset(False)

    def step(self, actions: Any) -> Tuple[List, List, bool, Dict]:
        obs, reward, done, info = super(FilteredGym, self).step(actions)
        b, o = self._team_sizes
        state = info.get("state")
        if state is not None:
            info["state"] = get_reduced_state(b, o, state)
        return obs[:b] + obs[3:3 + o], reward[:b] + reward[3:3 + o], done, info


if __name__ == '__main__':
    env = FilteredGym(1, 2, game_speed=1)

    while True:
        done = False
        env.reset()
        while not done:
            act = [env.action_space.sample() for _ in range(3)]
            obs, rew, done, info = env.step(act)
