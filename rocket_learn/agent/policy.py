from abc import ABC, abstractmethod

from torch import nn


class Policy(nn.Module, ABC):
    def __init__(self, identifier, rating, deterministic=False):
        super().__init__(identifier, rating)
        self.deterministic = deterministic

    @abstractmethod
    def forward(self, *args, **kwargs): raise NotImplementedError

    @abstractmethod
    def get_action_distribution(self, obs): raise NotImplementedError

    @staticmethod
    @abstractmethod
    def sample_action(distribution, deterministic=None): raise NotImplementedError

    @staticmethod
    @abstractmethod
    def log_prob(distribution, selected_action): raise NotImplementedError

    @staticmethod
    @abstractmethod
    def entropy(distribution, selected_action): raise NotImplementedError

    # @abstractmethod
    # def env_compatible(self, action): raise NotImplementedError
