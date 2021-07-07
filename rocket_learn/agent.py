from abc import abstractmethod, ABC
from typing import Tuple

import numpy as np


class BaseAgent(ABC):
    @abstractmethod
    def get_actions(self, observation, deterministic=False) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_action_with_log_prob(self, observation) -> Tuple[np.ndarray, float]:
        raise NotImplementedError

    @abstractmethod
    def set_model_params(self, params) -> None:
        raise NotImplementedError


class ActorCriticAgent(BaseAgent, ABC):
    @abstractmethod
    def forward_actor_critic(self, obs) -> Tuple:
        raise NotImplementedError

    def forward_actor(self, obs):
        return self.forward_actor_critic(obs)[0]

    def critic_actor(self, obs):
        return self.forward_actor_critic(obs)[1]
