from abc import ABC, abstractmethod
from typing import Iterator

from rocket_learn.experience_buffer import ExperienceBuffer


class BaseRolloutGenerator(ABC):
    @abstractmethod
    def generate_rollouts(self) -> Iterator[ExperienceBuffer]:
        raise NotImplementedError

    @abstractmethod
    def update_parameters(self, new_params):
        raise NotImplementedError
