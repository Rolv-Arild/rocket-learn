from typing import Tuple

import numpy as np
import torch as th
from torch.distributions import Categorical

from rocket_learn.Agents.BaseAgent import BaseAgent


class RandomAgent(BaseAgent):
    """Does softmax using biases alone"""
    def __init__(self, throttle=(0, 1, 2), steer=(0, 1, 0), pitch=(0, 0, 0), yaw=(0, 0, 0),
                 roll=(0, 0, 0), jump=(3, 0), boost=(0, 0), handbrake=(3, 0)):
        super().__init__()
        self.distributions = [
            Categorical(logits=th.as_tensor(logits).float())
            for logits in (throttle, steer, pitch, yaw, roll, jump, boost, handbrake)
        ]

    def get_actions(self, observation, deterministic=False) -> np.ndarray:
        actions = np.stack([dist.sample() for dist in self.distributions])
        return actions

    def get_action_with_log_prob(self, observation) -> Tuple[np.ndarray, float]:
        pass

    def set_model_params(self, params) -> None:
        pass

    def forward_actor_critic(self, obs):
        pass

    def get_model_params(self):
        pass