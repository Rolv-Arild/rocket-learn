from dataclasses import dataclass
from typing import Union, Any, Tuple

from rlgym.utils.gamestates import GameState


@dataclass(frozen=True)
class Scoreboard:
    blue: int
    orange: int
    timer: float  # Seconds remaining
    is_overtime: bool


class CustomObjectLogic:
    def __init__(self):
        pass

    def reset(self, initial_state: GameState):
        raise NotImplementedError

    def step(self, state: GameState) -> Any:
        raise NotImplementedError

    def done(self, state: GameState) -> Tuple[bool, bool]:  # Terminated, Truncated
        raise NotImplementedError


class ScoreboardLogic(CustomObjectLogic):
    def __init__(self, blue: int, orange: int, ticks_left: Union[int, float]):
        super().__init__()
        self.blue = blue
        self.orange = orange
        self.ticks_left = ticks_left
        self._scoreboard = None

    def reset(self, initial_state: GameState):
        raise NotImplementedError

    def step(self, state: GameState) -> Scoreboard:
        raise NotImplementedError

    def done(self, state: GameState) -> Tuple[bool, bool]:
        raise NotImplementedError

    @property
    def scoreboard(self):
        return self._scoreboard
