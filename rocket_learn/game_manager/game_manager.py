from abc import ABC
from typing import Dict, Tuple, Any, List

import numpy as np
from pettingzoo import ParallelEnv

from rocket_learn.experience_buffer import ExperienceBuffer


class GameManager(ABC):
    ROLLOUT = 0
    EVAL = 1
    SHOW = 2

    def __init__(self, env: ParallelEnv):
        self.env = env

    def generate_matchup(self) -> Tuple[Any, int]:
        raise NotImplementedError

    def rollout(self, matchup: Any):
        raise NotImplementedError

    def evaluate(self, matchup: Any):
        raise NotImplementedError

    def show(self, matchup: Any):
        raise NotImplementedError

    def run(self):
        while True:
            matchup, matchup_type = self.generate_matchup()
            if matchup_type == GameManager.SHOW:
                self.show(matchup)
            elif matchup_type == GameManager.ROLLOUT:
                self.evaluate(matchup)
            elif matchup_type == GameManager.EVAL:
                self.show(matchup)
            else:
                raise ValueError("Invalid matchup type")
