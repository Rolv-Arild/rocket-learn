import os
from typing import Generator

import torch.distributions
from torch.nn import Module

import rlgym
from rlgym.envs import Match
from rlgym.gym import Gym
from rocket_learn.experience_buffer import ExperienceBuffer

from rocket_learn.rollout_generator.base_rollout_generator import BaseRolloutGenerator


class SimpleRolloutGenerator(BaseRolloutGenerator):
    def __init__(self, net: Module, **make_args):
        self.env = rlgym.make(**make_args)
        self.net = net
        self.n_agents = self.env._match.agents

    def generate_rollouts(self) -> Generator:
        while True:
            observations = self.env.reset()  # Do we need to add this to buffer?
            done = False
            rollouts = [
                ExperienceBuffer()
                for _ in range(self.n_agents)
            ]

            while not done:
                # TODO require returning either: -torch.distributions.Distribution or -(selected_action,prob) tuple
                actions = [self.env(obs) for obs in observations]
                observations, rewards, done, info = self.env.step(actions)

            yield rollouts

    def update_parameters(self, new_params):
        self.net.load_state_dict(new_params)
