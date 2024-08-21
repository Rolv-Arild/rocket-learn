from typing import Dict, Any, List

import torch
from rlgym.api import AgentID, ObsType, ActionType, StateType
from rlgym.api.config import ObsBuilder, ActionParser, RewardFunction
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.modules import ProbabilisticActor

from rocket_learn.agent.config_agent import ConfigAgent


class TorchPPOAgent(ConfigAgent):
    def __init__(self, actor: ProbabilisticActor,
                 obs_builder: ObsBuilder,
                 action_parser: ActionParser,
                 reward_function: RewardFunction):
        super().__init__(obs_builder, action_parser, reward_function)
        self.actor = actor.eval()
        self.experience_buffers = {}

    def reset(self, agents: List[AgentID], initial_state: StateType, shared_info: Dict[str, Any]):
        super().reset(agents, initial_state, shared_info)
        self.experience_buffers = {k: TensorDictReplayBuffer(storage=LazyTensorStorage(max_size=1000))
                                   for k in agents}

    @torch.no_grad()
    def infer(self, obs: Dict[AgentID, ObsType], is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
              shared_info: Dict[str, Any]) -> Dict[AgentID, ActionType]:
        agents, observations = zip(*obs.items())
        agents = list(agents)
        observations = list(observations)
        output = self.actor(observations)

        actions = output["action"].cpu().numpy()

        for agent in agents:
            exp = self.experience_buffers[agent]
            term = is_terminated[agent]
            exp.add(TensorDict({
                "obs": observations[agent],
                "action": actions[agent],
                "reward": self.rewards[-1],
                "done": term or is_truncated[agent],
                "terminated": term
            }))

        return {ag: ac for ag, ac in zip(agents, actions)}
