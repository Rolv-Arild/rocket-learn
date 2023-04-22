from typing import Any, Set, Dict

import numpy as np
from redis import Redis
from rlgym.utils import ObsBuilder, RewardFunction
from rlgym.utils.action_parsers import ActionParser
from rlgym.utils.gamestates import GameState

from rocket_learn.agent.policy import Policy
from rocket_learn.agent.redis_agent import RedisAgent


class ConfigAgent(RedisAgent):
    def __init__(self, policy: Policy, redis: Redis,
                 obs_builder: ObsBuilder, reward_function: RewardFunction, action_parser: ActionParser,
                 send_obs: bool = True, send_states: bool = True):
        super().__init__(policy, redis, send_obs, send_states)
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

    def reset(self, initial_state: Any, agents: Set[str]):
        self.obs_builder.reset(initial_state)
        self.reward_function.reset(initial_state)
        super(ConfigAgent, self).reset(initial_state, agents)

    def end(self, final_state: Any, truncated: Dict[str, bool]):
        raise NotImplementedError
