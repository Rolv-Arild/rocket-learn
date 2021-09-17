from typing import Optional, List

import numpy as np
import torch as th
from torch import nn
from torch.distributions import Categorical

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

        return [Categorical(logits=logit) for logit in logits]

    def sample_action(
            self,
            distribution: List[Categorical],
            deterministic=False
    ):
        if deterministic:
            action_indices = th.stack([th.argmax(dist.logits) for dist in distribution])
        else:
            action_indices = th.stack([dist.sample() for dist in distribution])

        return action_indices

    def log_prob(self, distribution: List[Categorical], selected_action):
        log_prob = th.stack(
            [dist.log_prob(action) for dist, action in zip(distribution, th.unbind(selected_action, dim=-1))],
            dim=-1).sum(dim=-1)
        return log_prob

    def entropy(self, distribution, selected_action):
        entropy = th.stack([dist.entropy() for dist in distribution], dim=1).sum(dim=1)
        return entropy

    def env_compatible(self, action):
        if isinstance(action, th.Tensor):
            action = action.numpy()
        return self.index_action_map[np.arange(len(self.index_action_map)), action]
