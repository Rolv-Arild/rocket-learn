from typing import Tuple

import numpy as np
import torch
import torch as th
from torch import nn
from torch.distributions import Categorical
import pickle

from rocket_learn.agent import BaseAgent


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

    # def set_model_params(self, params):
    #     self.distributions = [
    #         Categorical(logits=th.as_tensor(logits).float())
    #         for logits in params
    #     ]
    #
    # def get_actions(self, observation, deterministic=False):
    #     actions = np.stack([dist.sample() for dist in self.distributions])
    #     return actions
    #
    # def get_log_prob(self, actions):
    #     return th.stack(
    #         [dist.log_prob(action) for dist, action in zip(self.distributions, th.unbind(actions, dim=1))], dim=1
    #     ).sum(dim=1)


# ** This should be in its own file or packaged with PPO **
class PPOAgent(BaseAgent):
    def __init__(self, actor: nn.Module, critic: nn.Module):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward_actor_critic(self, obs):
        return self.forward_actor(obs), self.forward_critic(obs)

    def forward_actor(self, obs):
        return self.actor(obs)

    def forward_critic(self, obs):
        return self.critic(obs)

    def get_model_params(self, params):
        return self.actor.state_dict(), self.critic.state_dict()

    def set_model_params(self, params) -> None:
        self.actor.load_state_dict(params[0])
        self.critic.load_state_dict(params[1])


class NoOpAgent(BaseAgent):
    def get_actions(self, observation, deterministic=False):
        return th.zeros((8,))

    def get_log_prob(self, actions):
        return 0

    def set_model_params(self, params):
        pass
