from abc import ABC, abstractmethod
from typing import Tuple
from torch import nn

from rocket_learn.agent.policy import Policy
from rocket_learn.agent.discrete_policy import DiscretePolicy

from rlgym.utils.gamestates import GameState


class PretrainedDiscretePolicy(DiscretePolicy):
    """
        A rocket-learn discrete policy pretrained and imported to be trained against

        :param agent_name: Name of agent, used to identify it during runs
        :param obs_builder_func: Function that will generate the correct observation from the gamestate
        :param net: policy net
        :param shape: action distribution shape
    """

    def __init__(self, agent_name, obs_builder_func, net: nn.Module, shape: Tuple[int, ...] = (3,) * 5 + (2,) * 3):
        super().__init__(net, shape)
        self.obs_builder_func = obs_builder_func
        self.agent_name = agent_name

    def build_obs(self, state: GameState):
        return self.obs_builder_func(state)


class HardcodedAgent(ABC):
    """
        An external bot prebuilt and imported to be trained against
    """

    @abstractmethod
    def act(self, state: GameState): raise NotImplementedError


class DemoJumpAgent(HardcodedAgent):
    def act(self, state: GameState):
        return [0, 0, 1, 0, 0]

