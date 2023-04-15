from abc import ABC
from typing import Dict

from pettingzoo import ParallelEnv

from rocket_learn.agent.agent import Agent
from rocket_learn.experience_buffer import ExperienceBuffer


class GameManager(ABC):
    ROLLOUT = 0
    EVAL = 1
    SHOW = 2

    def __init__(self, env: ParallelEnv):
        self.env = env

    def generate_matchup(self) -> (Dict[str, Agent], int):
        raise NotImplementedError

    def rollout(self, agent_policy: Dict[str, Agent]) -> Dict[str, ExperienceBuffer]:
        raise NotImplementedError

    def evaluate(self, agent_policy: Dict[str, Agent]) -> int:
        raise NotImplementedError

    def show(self, agent_policy: Dict[str, Agent]):
        raise NotImplementedError
