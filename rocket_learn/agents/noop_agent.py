import torch as th

from rocket_learn.Agents.BaseAgent import BaseAgent
import numpy as np


class NoOpAgent(BaseAgent):
    def get_actions(self, observation, deterministic=False) -> np.ndarray:
        return th.zeros((8,)).numpy()

    def get_log_prob(self, actions):
        return 0

    def set_model_params(self, params):
        pass

    def forward_actor_critic(self, obs):
        pass

    def get_model_params(self):
        pass
