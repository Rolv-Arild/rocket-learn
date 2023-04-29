from abc import ABC
from typing import Any, Set, Dict, Tuple

import numpy as np
import torch
from rlgym.utils.common_values import BLUE_TEAM
from rlgym.utils.gamestates import GameState

from rocket_learn.agent.policy import Policy
from rocket_learn.agent.rocket_league_agent import RocketLeagueAgent
from rocket_learn.utils.experience_buffer import ExperienceBuffer
from rocket_learn.scoreboard.scoreboard_logic import Scoreboard


class TorchAgent(RocketLeagueAgent, ABC):
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

    def reset(self, initial_state: GameState, agents: Set[str]):
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
        self._experience_buffers = {c: ExperienceBuffer() for c in agents}

    def run_model(self, all_obs):
        dists = self.policy.get_action_distribution(all_obs)
        action_indices = self.policy.sample_action(dists)
        log_probs = self.policy.log_prob(dists, action_indices)

        return action_indices, log_probs

    def act(self, agents_observations: Dict[str, Tuple[GameState, Scoreboard]]) -> Dict[str, np.ndarray]:
        cars = list(agents_observations.keys())
        # The assumption is that all cars will share the same object
        state = next(iter(agents_observations.values()))

        all_obs = self.build_observations(state, cars)
        all_rewards = self.assign_rewards(state, cars)

        if isinstance(all_obs[0], tuple):
            all_obs = tuple(
                torch.from_numpy(np.vstack(o for o in zip(*all_obs))).float()
            )
        else:
            all_obs = torch.from_numpy(np.vstack(all_obs)).float()

        action_indices, log_probs = self.run_model(all_obs)

        actions = self.parse_actions(action_indices, state)

        self._previous_actions = dict(zip(cars, actions))

        for i, car in enumerate(cars):
            exp_buffer: ExperienceBuffer = self._experience_buffers[car]
            exp_buffer.add_step(all_obs[i], action_indices[i], all_rewards[i], False, False, log_probs[i], None)

        return dict(zip(cars, actions))

    def end(self, final_state: GameState, truncated: Dict[str, bool]):
        for car, exp_buffer in self._experience_buffers.items():
            exp_buffer: ExperienceBuffer
            if truncated[car]:
                exp_buffer.truncated[-1] = True
            else:
                exp_buffer.terminated[-1] = True
