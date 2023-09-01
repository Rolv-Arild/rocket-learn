from abc import ABC
from typing import Tuple, Any, Union, Dict

from rlgym.api import RLGym


class GameManager(ABC):
    ROLLOUT = 0
    EVAL = 1
    SHOW = 2

    def __init__(self, envs: Union[Dict[int, RLGym], RLGym]):
        if isinstance(envs, RLGym):
            envs = {i: envs for i in (self.ROLLOUT, self.EVAL, self.SHOW)}
        self.envs = envs

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
