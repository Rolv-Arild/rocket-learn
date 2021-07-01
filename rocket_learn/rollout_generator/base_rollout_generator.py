from abc import ABC, abstractmethod
from typing import Generator


class BaseRolloutGenerator(ABC):
    @abstractmethod
    def generate_rollouts(self) -> Generator:
        raise NotImplementedError

    @abstractmethod
    def update_parameters(self, new_params):
        raise NotImplementedError
