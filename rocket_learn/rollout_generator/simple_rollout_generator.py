from typing import Generator

from torch.nn import Module

import rlgym
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.rollout_generator.base_rollout_generator import BaseRolloutGenerator
from rocket_learn.utils.util import generate_episode


class SimpleRolloutGenerator(BaseRolloutGenerator):
    def __init__(self, net: Module, **make_args):
        self.env = rlgym.make(**make_args)
        self.net = net
        self.n_agents = self.env._match.agents

    def generate_rollouts(self) -> Generator[ExperienceBuffer]:
        while True:
            rollouts = generate_episode(self.env, [self.net] * self.n_agents)

            yield from rollouts

    def update_parameters(self, new_params):
        self.net.load_state_dict(new_params)
