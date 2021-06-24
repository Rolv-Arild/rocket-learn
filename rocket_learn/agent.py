from abc import abstractmethod, ABC


class BaseAgent(ABC):
    @abstractmethod
    def get_actions(self, observation, deterministic=False):
        raise NotImplementedError

    @abstractmethod
    def get_log_prob(self, actions):
        raise NotImplementedError

    @abstractmethod
    def set_model_params(self, params):
        raise NotImplementedError


