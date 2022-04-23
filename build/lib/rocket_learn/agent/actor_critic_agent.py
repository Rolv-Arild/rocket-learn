import torch as th
from torch import nn

from rocket_learn.agent.policy import Policy


class ActorCriticAgent(nn.Module):
    def __init__(self, actor: Policy, critic: nn.Module, optimizer: th.optim.Optimizer):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        # self.algo = ?
        # TODO self.shared =

    def forward(self, *args, **kwargs):
        return self.actor(*args, **kwargs), self.critic(*args, **kwargs)
