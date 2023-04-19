from typing import Optional, Tuple, Dict, Union, List, Any

import gymnasium
import numpy as np
import rlgym
from pettingzoo.utils.env import ObsDict, ActionDict
from rlgym.gamelaunch import LaunchPreference
from rlgym.gym import Gym
from rlgym.utils import StateSetter, RewardFunction, ObsBuilder, TerminalCondition
from rlgym.utils.action_parsers import ActionParser, DefaultAction
from rlgym.utils.common_values import BLUE_TEAM
from rlgym.utils.gamestates import GameState, PlayerData

from rocket_learn.envs.rocket_league import RocketLeague
from rocket_learn.utils.truncation import TerminalTruncatedCondition
from rocket_learn.utils.gamestate_encoding import encode_gamestate


class GameStateObs(ObsBuilder):
    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        return state


class NoReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0


class RLGym(RocketLeague):
    def __init__(self,
                 tick_skip,
                 terminal_conditions,
                 state_setter,
                 # By default, the Agent class parses actions, builds observations and assigns rewards
                 action_parser=DefaultAction(),
                 obs_builder=GameStateObs(),
                 reward_fn=NoReward(),
                 game_speed=100,
                 spawn_opponents=True,
                 team_size=3,
                 gravity=1,
                 boost_consumption=1,
                 launch_preference=LaunchPreference.EPIC,
                 force_paging=False,
                 auto_minimize=True):
        super().__init__(blue=env._match._team_size, orange=env._match._team_size)  # noqa
        self._env = rlgym.make(game_speed=game_speed, tick_skip=tick_skip, spawn_opponents=spawn_opponents,
                               team_size=team_size, gravity=gravity, boost_consumption=boost_consumption,
                               terminal_conditions=terminal_conditions, reward_fn=reward_fn, obs_builder=obs_builder,
                               action_parser=action_parser, state_setter=state_setter,
                               launch_preference=launch_preference, use_injector=True, force_paging=force_paging,
                               raise_on_crash=True, auto_minimize=auto_minimize)
        self._state = None

    @classmethod
    def make(cls, *args, **kwargs):
        return cls(rlgym.make(*args, **kwargs))

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> ObsDict:
        self._env.update_settings(game_speed=options.get("game_speed", None),
                                  gravity=options.get("gravity", None),
                                  boost_consumption=options.get("boost_consumption", None))
        obs, info = self._env.reset(return_info=True)
        state: GameState = info["state"]

        b = o = 0
        player_ids = []
        for player in state.players:
            if player.team_num == BLUE_TEAM:
                player_ids.append(f"blue-{b}")
                b += 1
            else:
                player_ids.append(f"orange-{o}")
                o += 1
        self.agents = player_ids

        obs = dict(zip(self.agents, obs))

        self._state = state

        return obs, info if return_info else obs  # noqa ObsDict does not account for info

    def seed(self, seed=None):
        pass

    def is_truncated(self, state):
        for condition in self._env._match._terminal_conditions:  # noqa
            if isinstance(condition, TerminalTruncatedCondition) \
                    and condition.is_truncated(state):
                return True
        return False

    def step(self, actions: ActionDict) -> Tuple[
        ObsDict, Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]
    ]:
        obs, reward, done, info = self._env.step([actions[agent] for agent in self.agents])

        state = info["state"]
        truncated = self.is_truncated(info["state"])

        self._state = state

        obs = dict(zip(self.agents, obs))
        reward = dict(zip(self.agents, reward))
        done = {a: done for a in self.agents}
        truncated = {a: truncated for a in self.agents}
        info = {a: info for a in self.agents}

        return obs, reward, done, truncated, info

    def state(self) -> np.ndarray:
        return encode_gamestate(self._state)

    def render(self) -> Union[None, np.ndarray, str, List]:
        return None

    def observation_space(self, agent: str) -> gymnasium.spaces.Space:
        space = self._env._match._obs_builder.get_obs_space()  # noqa
        if isinstance(space, dict):
            return space[agent]
        return space

    def action_space(self, agent: str) -> gymnasium.spaces.Space:
        space = self._env._match._action_parser.get_action_space()  # noqa
        if isinstance(space, dict):
            return space[agent]
        return space

    @property
    def game_speed(self):
        return self._env._match._gamespeed  # noqa

    @game_speed.setter
    def game_speed(self, value):
        self._env.update_settings(game_speed=value)

    @property
    def gravity(self):
        return self._env._match._gravity  # noqa

    @gravity.setter
    def gravity(self, value):
        self._env.update_settings(gravity=value)

    @property
    def boost_consumption(self):
        return self._env._match._boost_consumption  # noqa

    @boost_consumption.setter
    def boost_consumption(self, value):
        self._env.update_settings(boost_consumption=value)

    @property
    def tick_skip(self):
        return self._env._match._tick_skip  # noqa

    @property
    def terminal_conditions(self):
        return self._env._match._terminal_conditions  # noqa

    @terminal_conditions.setter
    def terminal_conditions(self, *terminal_condition: TerminalCondition):
        self._env._match._terminal_conditions = list(terminal_condition)  # noqa

    @property
    def reward_fn(self):
        return self._env._match._reward_fn  # noqa

    @reward_fn.setter
    def reward_fn(self, reward_fn: RewardFunction):
        self._env._match._reward_fn = reward_fn  # noqa

    @property
    def obs_builder(self):
        return self._env._match._obs_builder  # noqa

    @obs_builder.setter
    def obs_builder(self, obs_builder: ObsBuilder):
        self._env._match._obs_builder = obs_builder  # noqa

    @property
    def action_parser(self):
        return self._env._match._action_parser  # noqa

    @action_parser.setter
    def action_parser(self, action_parser: ActionParser):
        self._env._match._action_parser = action_parser  # noqa

    @property
    def state_setter(self):
        return self._env._match._state_setter  # noqa

    @state_setter.setter
    def state_setter(self, state_setter: StateSetter):
        self._env._match._state_setter = state_setter  # noqa

    @property
    def spawn_opponents(self):
        return self._env._match._spawn_opponents  # noqa

    @property
    def team_size(self):
        return self._env._match._team_size  # noqa
