from typing import Optional, List

import numpy as np
import torch as th
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F

from rocket_learn.agent.policy import Policy


class ContinuousPolicy(Policy):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        
        self.min_std = 0.1
        self.max_std = 0.65

    def forward(self, obs):
        model_out = self.net(obs)
        return model_out

    def get_action_distribution(self, obs):
        if isinstance(obs, np.ndarray):
            obs = th.from_numpy(obs).float()
        elif isinstance(obs, tuple):
            obs = tuple(o if isinstance(o, th.Tensor) else th.from_numpy(o).float() for o in obs)

        model_out = self(obs)
        
        mean, std = self._map_policy_to_continuous_action(model_out)
        return Normal(loc=mean, scale=std)

    def sample_action(self, distribution: Normal, deterministic=False):
        if deterministic:
            action = distribution.mean
        else:
            action = distribution.sample()

        return action

    def log_prob(self, distribution: Normal, selected_action):
        log_prob = self._logpdf(selected_action, distribution.loc, distribution.scale).sum(dim=-1)
        return log_prob

    def entropy(self, distribution: Normal, selected_action):
        entropy = distribution.entropy().sum(dim=-1)
        return entropy

    def env_compatible(self, action):
        return True
        
    def _logpdf(self, x, mean, std):
        msq = mean*mean
        ssq = std*std
        xsq = x*x

        term1 = -msq/(2*ssq)
        term2 = mean*x/ssq
        term3 = -xsq/(2*ssq)
        term4 = torch.log(1/torch.sqrt(2*np.pi*ssq))
        return term1 + term2 + term3 + term4
        
    def _apply_affine_map(self, value, from_min, from_max, to_min, to_max):
        if from_max == from_min or to_max == to_min:
            return to_min

        mapped = (value - from_min) * (to_max - to_min) / (from_max - from_min)
        mapped += to_min

        return mapped
        
    def _map_policy_to_continuous_action(self, policy_output):
        n = policy_output.shape[-1]//2
        if len(policy_output.shape) == 1:
            mean = policy_output[:n]
            std = policy_output[n:]

        else:
            mean = policy_output[:, :n]
            std = policy_output[:, n:]

        std = self._apply_affine_map(std, -1, 1, self.min_std, self.max_std)
        return mean, std
