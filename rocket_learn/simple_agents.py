import numpy as np
import torch as th
from torch.distributions import Categorical

from agent import BaseAgent

class RandomAgent(BaseAgent):
    """Does softmax using biases alone"""

    def __init__(self, throttle=(0, 1, 2), steer=(0, 1, 0), pitch=(0, 0, 0), yaw=(0, 0, 0),
                 roll=(0, 0, 0), jump=(3, 0), boost=(0, 0), handbrake=(3, 0)):
        super().__init__()
        self.distributions = [
            Categorical(logits=th.as_tensor(logits).float())
            for logits in (throttle, steer, pitch, yaw, roll, jump, boost, handbrake)
        ]

    def set_model_params(self, params):
        self.distributions = [
            Categorical(logits=th.as_tensor(logits).float())
            for logits in params
        ]

    def get_actions(self, observation, deterministic=False):
        actions = np.stack([dist.sample() for dist in self.distributions])
        return actions

    def get_log_prob(self, actions):
        return th.stack(
            [dist.log_prob(action) for dist, action in zip(self.distributions, th.unbind(actions, dim=1))], dim=1
        ).sum(dim=1)


# ** This should be in its own file or packaged with PPO **
class PPOAgent(BaseAgent):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def get_actions(self, observation, deterministic=False):
        return self.actor(observation)

    def get_log_prob(self, actions):
        return 0

    def set_model_params(self, params):
        pass

    def serialize(self):
        # **TODO **
        nets = [self.actor, self.critic]
        return pickle(self)

    def unserialize(self, pickle_data):
        # **TODO **
        return

class NoOpAgent(BaseAgent):
    def get_actions(self, observation, deterministic=False):
        return th.zeros((8,))

    def get_log_prob(self, actions):
        return 0

    def set_model_params(self, params):
        pass
