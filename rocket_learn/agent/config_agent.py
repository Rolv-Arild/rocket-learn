from typing import Dict, Any, List

from rocket_learn.agent.agent import Agent
from rlgym.api.config import ObsBuilder, ActionParser, RewardFunction
from rlgym.api import StateType, AgentID, EngineActionType, ObsType, ActionType


class ConfigAgent(Agent):
    def __init__(self, obs_builder: ObsBuilder, action_parser: ActionParser, reward_function: RewardFunction):
        self.obs_builder = obs_builder
        self.action_parser = action_parser
        self.reward_function = reward_function
        self.done_agents = set()
        self.rewards = []

    def reset(self, agents: List[AgentID], initial_state: StateType, shared_info: Dict[str, Any]):
        self.obs_builder.reset(agents, initial_state, shared_info)
        self.action_parser.reset(agents, initial_state, shared_info)
        self.reward_function.reset(agents, initial_state, shared_info)
        self.rewards.clear()

    def infer(self, obs: Dict[AgentID, ObsType], is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
              shared_info: Dict[str, Any]) -> Dict[AgentID, ActionType]:
        raise NotImplementedError

    def act(self, agents: List[AgentID], state: StateType,
            is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
            shared_info: Dict[str, Any]) -> Dict[AgentID, EngineActionType]:
        agents = [agent for agent in agents if agent not in self.done_agents]
        rewards = self.reward_function.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        self.rewards.append(rewards)

        for agent in agents:
            if is_terminated[agent] or is_truncated[agent]:
                self.done_agents.add(agent)
                agents = [a for a in agents if a != agent]  # Slightly inefficient but only done once per agent per ep

        obs = self.obs_builder.build_obs(agents, state, shared_info)
        actions = self.infer(obs, is_terminated, is_truncated, shared_info)
        engine_actions = self.action_parser.parse_actions(actions, state, shared_info)
        return engine_actions
