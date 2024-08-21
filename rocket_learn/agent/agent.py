from abc import abstractmethod
from typing import Generic, List, Dict, Any

from rlgym.api import AgentID, StateType, EngineActionType


class Agent(Generic[AgentID, StateType, EngineActionType]):
    @abstractmethod
    def reset(self, agents: List[AgentID], initial_state: StateType, shared_info: Dict[str, Any]):
        """
        Reset the agent to prepare for a new episode.

        :param agents: List of agent ids
        :param initial_state: Initial state of the environment
        :param shared_info: Shared information between agents
        """
        raise NotImplementedError

    @abstractmethod
    def act(self, agents: List[AgentID], state: StateType,
            is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
            shared_info: Dict[str, Any]) -> Dict[AgentID, EngineActionType]:
        """
        Generate actions for specified agents given the current state of the environment and shared information.

        :param agents: List of agent ids
        :param state: Current state of the environment
        :param is_terminated: Dictionary of agent ids to whether the agent has terminated
        :param is_truncated: Dictionary of agent ids to whether the episode was truncated
        :param shared_info: Shared information between agents
        :return: Dictionary of agent ids to actions
        """
        raise NotImplementedError
