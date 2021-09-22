from typing import Optional, List

import numpy as np
import torch as th
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from rocket_learn.agent.policy import Policy


class DiscretePolicy(Policy):
    def __init__(self, net: nn.Module, index_action_map: Optional[np.ndarray] = None):
        super().__init__()
        self.net = net
        if index_action_map is None:
            self.index_action_map = np.array([
                [-1., 0., 1.],
                [-1., 0., 1.],
                [-1., 0., 1.],
                [-1., 0., 1.],
                [-1., 0., 1.],
                [0., 1., np.nan],
                [0., 1., np.nan],
                [0., 1., np.nan]
            ])
        else:
            self.index_action_map = index_action_map

    def forward(self, obs):
        logits = self.net(obs)
        return logits

    def get_action_distribution(self, obs):
        if isinstance(obs, np.ndarray):
            obs = th.from_numpy(obs).float()
        elif isinstance(obs, tuple):
            obs = tuple(o if isinstance(o, th.Tensor) else th.from_numpy(o).float() for o in obs)

        logits = self(obs)

        triplets = th.stack(logits[:5])
        duets = F.pad(th.stack(logits[5:]), pad=(0, 1), value=float("-inf"))
        logits = th.cat((triplets, duets)).swapdims(0, 1).squeeze()

        return Categorical(logits=logits)

    def sample_action(
            self,
            distribution: Categorical,
            deterministic=False
    ):
        if deterministic:
            action_indices = th.argmax(distribution.logits)
        else:
            action_indices = distribution.sample()

        return action_indices

    def log_prob(self, distribution: Categorical, selected_action):
        log_prob = distribution.log_prob(selected_action).sum(dim=-1)
        return log_prob

    def entropy(self, distribution: Categorical, selected_action):
        entropy = distribution.entropy().sum(dim=-1)
        return entropy

    def env_compatible(self, action):
        if isinstance(action, th.Tensor):
            action = action.numpy()
        return self.index_action_map[np.arange(len(self.index_action_map)), action]
