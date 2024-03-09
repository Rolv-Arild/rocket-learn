from typing import Iterator

import rlgym
from rocket_learn.agent.policy import Policy
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.rollout_generator.base_rollout_generator import BaseRolloutGenerator
from rocket_learn.utils.generate_episode import generate_episode


class SimpleRolloutGenerator(BaseRolloutGenerator):
    def __init__(self, policy: Policy, **make_args):
        self.env = rlgym.make(**make_args)
        self.policy = policy
        self.n_agents = self.env._match.agents

    def generate_rollouts(self) -> Iterator[ExperienceBuffer]:
        while True:
            # TODO: need to add selfplay agent here?
            rollouts, result = generate_episode(self.env, [self.policy, list(range(self.n_agents))])

            yield from rollouts

    def update_parameters(self, new_params):
        self.policy = new_params
