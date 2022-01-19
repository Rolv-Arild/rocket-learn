from abc import ABC, abstractmethod

from torch import nn


class Policy(nn.Module, ABC):
    @abstractmethod
    def forward(self, *args, **kwargs): raise NotImplementedError

    @abstractmethod
    def jit_compile_net(self, obs): raise NotImplementedError

    @abstractmethod
    def get_action_distribution(self, obs): raise NotImplementedError

    @staticmethod
    @abstractmethod
    def sample_action(distribution, deterministic=False): raise NotImplementedError

    @staticmethod
    @abstractmethod
    def log_prob(distribution, selected_action): raise NotImplementedError

    @staticmethod
    @abstractmethod
    def entropy(distribution, selected_action): raise NotImplementedError

    @abstractmethod
    def env_compatible(self, action): raise NotImplementedError
