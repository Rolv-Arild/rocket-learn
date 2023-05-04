from typing import Tuple

from rlgym.utils.gamestates import GameState

from rocket_learn.scoreboard.scoreboard_logic import ScoreboardLogic, Scoreboard


class NullScoreboardLogic(ScoreboardLogic):
    def __init__(self):
        super().__init__(0, 0, float("inf"))

    def reset(self, initial_state: GameState):
        pass

    def step(self, state: GameState) -> Scoreboard:
        return Scoreboard(0, 0, float("inf"), True)

    def done(self, state: GameState) -> Tuple[bool, bool]:
        pass
