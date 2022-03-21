from abc import ABC, abstractmethod
from typing import Tuple
from torch import nn

from rocket_learn.agent.discrete_policy import DiscretePolicy

from rlgym.utils.gamestates import GameState

class HardcodedAgent(ABC):
    """
        An external bot prebuilt and imported to be trained against
    """

    @abstractmethod
    def act(self, state: GameState): raise NotImplementedError


class PretrainedDiscretePolicy(DiscretePolicy, HardcodedAgent):
    """
        A rocket-learn discrete policy pretrained and imported to be trained against

        :param obs_builder_func: Function that will generate the correct observation from the gamestate
        :param net: policy net
        :param shape: action distribution shape
    """

    def __init__(self, obs_builder_func, net: nn.Module, shape: Tuple[int, ...] = (3,) * 5 + (2,) * 3):
        super().__init__(net, shape)
        self.obs_builder_func = obs_builder_func

    def act(self, state: GameState):
        obs = self.obs_builder_func(state)
        dist = self.get_action_distribution(obs)
        action_indices = self.sample_action(dist, deterministic=False)
        actions = self.env_compatible(action_indices)

        return actions


class DemoDriveAgent(HardcodedAgent):
    def act(self, state: GameState):
        return [2, 1, 1, 0, 0, 0, 0, 0]


class DemoKBMDriveAgent(HardcodedAgent):
    def act(self, state: GameState):
        return [2, 1, 0, 0, 0]

