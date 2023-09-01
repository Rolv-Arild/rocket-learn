import math
from dataclasses import dataclass
from typing import Union, Any, Dict

from rlgym.rocket_league.api import GameState

from rocket_learn.custom_objects.custom_object_logic import CustomObjectLogic


@dataclass(frozen=True)
class Scoreboard:
    blue: int
    orange: int
    timer: float  # Seconds remaining

    @property
    def is_overtime(self) -> bool:
        return self.timer > 0 and math.isinf(self.timer)

    @property
    def is_finished(self) -> bool:
        return self.timer < 0 and math.isinf(self.timer)


class ScoreboardLogic(CustomObjectLogic):
    def __init__(self, blue: int, orange: int, ticks_left: Union[int, float]):
        super().__init__()
        self.blue = blue
        self.orange = orange
        self.ticks_left = ticks_left

    def reset(self, initial_state: GameState, shared_info: Dict[str, Any]):
        raise NotImplementedError

    def step(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        raise NotImplementedError
