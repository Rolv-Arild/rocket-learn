from typing import Generator, Iterator
import rlgym
from rocket_learn.agents.base_agent import BaseAgent
from rocket_learn.utils.experiencebuffer import ExperienceBuffer
from rocket_learn.rollout_generators.base_rolloutgenerator import BaseRolloutGenerator
from rocket_learn.utils.util import generate_episode


class SimpleRolloutGenerator(BaseRolloutGenerator):
    def __init__(self, agent: BaseAgent, **make_args):
        self.env = rlgym.make(**make_args)
        self.agent = agent
        self.n_agents = self.env._match.agents

    def generate_rollouts(self) -> Iterator[ExperienceBuffer]:
        while True:
            #TODO: need to add selfplay agent here?
            rollouts, result = generate_episode(self.env, [self.agent] * self.n_agents)

            yield from rollouts

    def update_parameters(self, new_params):
        self.agent.set_model_params(new_params)
