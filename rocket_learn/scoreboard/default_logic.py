import math
import random
from typing import Tuple, Literal

from numpy.random import poisson
from rlgym.utils.gamestates import GameState

from rocket_learn.scoreboard.scoreboard_logic import ScoreboardLogic, Scoreboard
from rocket_learn.scoreboard.util import GOALS_PER_MIN, SECONDS_PER_MINUTE, TICKS_PER_SECOND


class DefaultScoreboardLogic(ScoreboardLogic):
    def __init__(self, tick_skip, reset_mode=Literal["random", "beginning", "continue"],
                 max_time_seconds=300):
        super().__init__(0, 0, max_time_seconds)
        self.reset_mode = reset_mode
        self.tick_skip = tick_skip
        self.max_time_seconds = max_time_seconds
        self.state = None
        self._scoreboard = None
        self._truncate = False
        self._terminate = False

    def reset(self, initial_state: GameState):
        self.state = initial_state
        players_per_team = len(initial_state.players) // 2
        if self.reset_mode == "random":
            gpm = GOALS_PER_MIN[players_per_team - 1]
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
            term, trunc = self.done()
            if term:
                self.blue = 0
                self.orange = 0
                self.ticks_left = self.max_time_seconds * TICKS_PER_SECOND
        else:
            raise ValueError("Invalid reset mode")

        is_overtime = self.ticks_left > 0 and math.isinf(self.ticks_left)

        self._scoreboard = Scoreboard(self.blue, self.orange,
                                      self.ticks_left * TICKS_PER_SECOND,
                                      is_overtime)

    def step(self, state: GameState) -> Scoreboard:
        if state != self.state:
            was_overtime = self.ticks_left > 0 and math.isinf(self.ticks_left)

            if state.ball.position[1] != 0:  # Don't count during kickoffs
                self.ticks_left = max(0, self.ticks_left - self.tick_skip)

            b, o = self.blue, self.orange
            changed = False
            if state.blue_score > self.state.blue_score:  # Check in case of crash
                b += state.blue_score - self.state.blue_score
                changed = True
            if state.orange_score > self.state.orange_score:
                o += state.orange_score - self.state.orange_score
                changed = True
            tied = b == o
            if self._scoreboard.is_overtime:
                if not tied:
                    self.ticks_left = float("-inf")  # Finished
            if self.ticks_left <= 0 and (state.ball.position[2] <= 110 or changed):
                if tied:
                    self.ticks_left = float("inf")  # Overtime
                else:
                    self.ticks_left = float("-inf")  # Finished
            self.blue = b
            self.orange = o

            self.state = state

            is_overtime = self.ticks_left > 0 and math.isinf(self.ticks_left)
            is_finished = self.ticks_left < 0 and math.isinf(self.ticks_left)

            self._truncate = is_overtime and not was_overtime
            self._terminate = is_finished

            self._scoreboard = Scoreboard(self.blue, self.orange,
                                          self.ticks_left * TICKS_PER_SECOND,
                                          is_overtime)

        return self._scoreboard

    def done(self) -> Tuple[bool, bool]:
        return self._terminate, self._truncate  # TODO how to reset when ball hits ground at 0:00?
