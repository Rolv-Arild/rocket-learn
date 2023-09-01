from typing import Dict, Any

from rlgym.rocket_league.api import GameState

from rocket_learn.custom_objects.scoreboard.scoreboard import ScoreboardLogic, Scoreboard


class GoldenGoalScoreboardLogic(ScoreboardLogic):
    def __init__(self):
        super().__init__(0, 0, float("inf"))

    def reset(self, initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def step(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        shared_info["scoreboard"] = Scoreboard(0, 0, float("inf"))
