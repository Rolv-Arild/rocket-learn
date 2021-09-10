from rocket_learn.agents.base_agent import BaseAgent
import torch
import numpy as np
from torch import nn
from typing import Optional
from torch.nn import Identity
import io


def _default_collate(observations):
    return torch.as_tensor(np.stack(observations)).float()


class PPOAgent(BaseAgent):
    def __init__(self, actor: nn.Module, critic: nn.Module, shared: Optional[nn.Module] = None, collate_fn=None):
        super().__init__()
        self.actor = actor
        self.critic = critic
        if shared is None:
            shared = Identity()
        self.shared = shared
        self.collate_fn = _default_collate if collate_fn is None else collate_fn

    def forward_actor_critic(self, obs):
        if self.shared is not None:
            obs = self.shared(obs)
        return self.actor(obs), self.critic(obs)

    def forward_actor(self, obs):
        if self.shared is not None:
            obs = self.shared(obs)
        return self.actor(obs)

    def forward_critic(self, obs):
        if self.shared is not None:
            obs = self.shared(obs)
        return self.critic(obs)

    def get_model_params(self):
        buf = io.BytesIO()
        torch.save([self.actor, self.critic, self.shared], buf)
        return buf

    def set_model_params(self, params) -> None:
        torch.load(params.read())