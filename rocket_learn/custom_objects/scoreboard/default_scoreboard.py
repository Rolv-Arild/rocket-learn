import random
from typing import Tuple, Literal, Dict, Any, Optional

from numpy.random import poisson
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BLUE_TEAM

from rocket_learn.custom_objects.scoreboard.scoreboard import ScoreboardLogic, Scoreboard
from rocket_learn.custom_objects.scoreboard.util import SECONDS_PER_MINUTE, TICKS_PER_SECOND, DEFAULT_GOALS_PER_MIN


class DefaultScoreboardLogic(ScoreboardLogic):
    def __init__(self, reset_mode=Literal["random", "beginning", "continue"],
                 max_time_seconds=300, goals_per_min: Optional[Dict[Tuple[int, int], float]] = None):
        super().__init__(0, 0, max_time_seconds)
        self.reset_mode = reset_mode
        self.max_time_seconds = max_time_seconds
        self.state = None
        self._scoreboard = None
        if goals_per_min is None:
            goals_per_min = DEFAULT_GOALS_PER_MIN
        self.goals_per_min = goals_per_min

    def reset(self, initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.state = initial_state
        players_per_team = len(initial_state.cars) // 2
        if self.reset_mode == "random":
            gpm = self.goals_per_min[(players_per_team, players_per_team)]
            mu_full = gpm * self.max_time_seconds / SECONDS_PER_MINUTE
            full_game = poisson(mu_full, size=2)
            if full_game[1] == full_game[0]:
                self.ticks_left = float("inf")  # Overtime
                b = o = full_game[0].item()
            else:
                max_ticks = self.max_time_seconds * TICKS_PER_SECOND
                self.ticks_left = random.randrange(0, max_ticks)
                seconds_spent = self.max_time_seconds - self.ticks_left / TICKS_PER_SECOND
                mu_spent = gpm * seconds_spent / SECONDS_PER_MINUTE
                b, o = poisson(mu_spent, size=2).tolist()
            self.blue = b
            self.orange = o
        elif self.reset_mode == "beginning":
            self.blue = 0
            self.orange = 0
            self.ticks_left = self.max_time_seconds * TICKS_PER_SECOND
        elif self.reset_mode == "continue":
            if self._scoreboard.is_finished:
                self.blue = 0
                self.orange = 0
                self.ticks_left = self.max_time_seconds * TICKS_PER_SECOND
        else:
            raise ValueError("Invalid reset mode")

        self._scoreboard = Scoreboard(self.blue, self.orange,
                                      self.ticks_left * TICKS_PER_SECOND)

    def step(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        if state != self.state:
            if state.ball.position[1] != 0:  # Don't count during kickoffs
                ticks_passed = state.tick_count - self.state.tick_count
                self.ticks_left = max(0, self.ticks_left - ticks_passed)

            b, o = self.blue, self.orange
            if state.goal_scored:
                if state.scoring_team == BLUE_TEAM:
                    b += 1
                else:
                    o += 1
            tied = b == o
            if self._scoreboard.is_overtime:
                if not tied:
                    self.ticks_left = float("-inf")  # Finished
            if self.ticks_left <= 0 and (state.ball.position[2] <= 110 or state.goal_scored):
                if tied:
                    self.ticks_left = float("inf")  # Overtime
                else:
                    self.ticks_left = float("-inf")  # Finished
            self.blue = b
            self.orange = o

            self.state = state

            self._scoreboard = Scoreboard(self.blue, self.orange,
                                          self.ticks_left * TICKS_PER_SECOND)
        shared_info["scoreboard"] = self._scoreboard
