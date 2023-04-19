from abc import ABC
from typing import Union, Any

import numpy as np
import torch
from redis import Redis
from rlgym.utils import ObsBuilder, RewardFunction
from rlgym.utils.action_parsers import ActionParser
from rlgym.utils.common_values import BLUE_TEAM
from rlgym.utils.gamestates import GameState
from trueskill import Rating

from rocket_learn.agent.policy import Policy
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.rollout_generator.redis.utils import _serialize, encode_buffers, ROLLOUTS, _unserialize_model, \
    OPPONENT_MODELS, VERSION_LATEST, MODEL_LATEST


class Agent:
    def __init__(self, is_multi: bool):
        self._is_multi = is_multi

    @property
    def is_multi(self):
        return self._is_multi

    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def act(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def end(self, *args, **kwargs):
        raise NotImplementedError


class RLAgent(Agent):
    def __init__(self, is_multi: bool):
        super().__init__(is_multi)

    def reset(self, initial_state: GameState):
        raise NotImplementedError

    def act(self, state: GameState, cars: Union[str, list[str]]):
        raise NotImplementedError

    def end(self, final_state: GameState):
        raise NotImplementedError


class TorchAgent(RLAgent, ABC):
    def __init__(self, policy: Policy):
        super().__init__(True)
        self.policy = policy
        self._previous_actions = {}
        self._car_to_index = {}
        self._experience_buffers = {}

    def build_observations(self, state: GameState, cars):
        raise NotImplementedError

    def assign_rewards(self, state: GameState, cars):
        raise NotImplementedError

    def parse_actions(self, actions: Any, state: GameState):
        raise NotImplementedError

    def reset(self, initial_state: GameState):
        car_to_index = {}
        b = o = 0
        for i, player in enumerate(initial_state.players):
            if player.team_num == BLUE_TEAM:
                player_car = f"blue-{b}"
                b += 1
            else:
                player_car = f"orange-{o}"
                o += 1
            car_to_index = {player_car: b + o - 1}
        self._car_to_index = car_to_index
        self._experience_buffers = {c: ExperienceBuffer() for c in car_to_index}

    def run_model(self, all_obs):
        dists = self.policy.get_action_distribution(all_obs)
        action_indices = self.policy.sample_action(dists)
        log_probs = self.policy.log_prob(dists, action_indices)

        return action_indices, log_probs

    def act(self, state: GameState, cars: Union[str, set[str]]) -> dict[str, np.ndarray]:
        if isinstance(cars, str):
            cars = {cars}

        cars = list(cars)

        all_obs = self.build_observations(state, cars)
        all_rewards = self.assign_rewards(state, cars)

        if isinstance(all_obs[0], tuple):
            all_obs = tuple(
                torch.from_numpy(np.stack(o for o in zip(*all_obs))).float()
            )
        else:
            all_obs = torch.from_numpy(np.stack(all_obs)).float()

        action_indices, log_probs = self.run_model(all_obs)

        actions = self.parse_actions(action_indices, state)

        self._previous_actions = dict(zip(cars, actions))

        for i, car in enumerate(cars):
            exp_buffer: ExperienceBuffer = self._experience_buffers[car]
            exp_buffer.add_step(all_obs[i], action_indices[i], all_rewards[i], False, False, log_probs[i], None)

        return dict(zip(cars, actions))


class RLAgentWithConfig(TorchAgent):
    def __init__(self, policy: Policy,
                 obs_builder: ObsBuilder, reward_function: RewardFunction, action_parser: ActionParser):
        super().__init__(policy)
        self.obs_builder = obs_builder
        self.reward_function = reward_function
        self.action_parser = action_parser

    def build_observations(self, state: GameState, cars):
        cars = list(cars)

        self.obs_builder.pre_step(state)
        self.reward_function.pre_step(state)

        all_obs = []

        for car in cars:
            idx = self._car_to_index[car]
            player = state.players[idx]

            prev_action = self._previous_actions.get(car, np.zeros(8))
            obs = self.obs_builder.build_obs(player, state, prev_action)
            all_obs.append(obs)

        return all_obs

    def assign_rewards(self, state: GameState, cars):
        self.obs_builder.pre_step(state)
        self.reward_function.pre_step(state)

        all_rewards = []

        for car in cars:
            idx = self._car_to_index[car]
            player = state.players[idx]

            prev_action = self._previous_actions.get(car, np.zeros(8))
            rew = self.reward_function.get_reward(player, state, prev_action)
            all_rewards.append(rew)

        return all_rewards

    def parse_actions(self, actions: Any, state: GameState):
        return self.action_parser.parse_actions(actions, state)

    def reset(self, initial_state: GameState):
        self.obs_builder.reset(initial_state)
        self.reward_function.reset(initial_state)
        super(RLAgentWithConfig, self).reset(initial_state)

    def end(self, final_state: GameState):
        raise NotImplementedError


def send_experience_buffers(redis: Redis, identifiers: list[str], experience_buffers: list[ExperienceBuffer],
                            send_obs: bool, send_states: bool):
    rollout_data = encode_buffers(experience_buffers,
                                  return_obs=send_obs,
                                  return_states=send_states,
                                  return_rewards=True)
    # sanity_check = decode_buffers(rollout_data, versions,
    #                               has_obs=False, has_states=True, has_rewards=True,
    #                               obs_build_factory=lambda: self.match._obs_builder,
    #                               rew_func_factory=lambda: self.match._reward_fn,
    #                               act_parse_factory=lambda: self.match._action_parser)
    rollout_bytes = _serialize((rollout_data, identifiers,
                                send_obs, send_states, True))

    # TODO async communication?

    n_items = redis.rpush(ROLLOUTS, rollout_bytes)
    if n_items >= 1000:
        print("Had to limit rollouts. Learner may have have crashed, or is overloaded")
        redis.ltrim(ROLLOUTS, -100, -1)


class RedisAgent(RLAgentWithConfig):
    def __init__(self, policy: Policy,
                 obs_builder: ObsBuilder, reward_function: RewardFunction, action_parser: ActionParser,
                 redis: Redis, send_obs: bool = True, send_states: bool = True):
        super().__init__(policy, obs_builder, reward_function, action_parser)
        self.redis = redis

        self.send_obs = send_obs
        self.send_states = send_states

    def _load_policy(self, identifier=None):
        if identifier is None:
            identifier = self._identifier
        agent, version, mode = identifier.split("-")
        if mode == "live":
            latest = self.redis.get(VERSION_LATEST)
            policy = _unserialize_model(self.redis.hget(MODEL_LATEST, agent))
        else:
            policy = _unserialize_model(self.redis.hget(OPPONENT_MODELS, identifier))
            if mode == "stochastic":
            elif mode == "deterministic":
        self.policy = policy

    @property
    def identifier(self):
        return self._identifier

    @identifier.setter
    def identifier(self, value):
        if self._identifier != value:
            self._load_policy(value)

    def end(self, final_state: GameState):
        buffers = list(self._experience_buffers.values())
        identifiers = [self.identifier] * len(buffers)
        send_experience_buffers(self.redis, identifiers, buffers, self.send_obs, self.send_states)

class LiveAgent(RedisAgent):


class FixedAgent(RedisAgent):

    def end(self, final_state: GameState):
        pass
