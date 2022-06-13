import math
import random

import numpy as np
from numpy.random import poisson
from rlgym.utils.common_values import BACK_WALL_Y, SIDE_WALL_X, GOAL_HEIGHT

from rlgym.utils.gamestates import GameState

TICKS_PER_SECOND = 120
SECONDS_PER_MINUTE = 60
GOALS_PER_MIN = (1, 0.6, 0.45)  # Stats from ballchasing, S14 GC (before SSL)


class Scoreboard:
    def __init__(self, random_resets=True, tick_skip=8, max_time_seconds=300, skip_warning=False):
        super().__init__()
        self.random_resets = random_resets
        self.tick_skip = tick_skip
        self.max_time_seconds = max_time_seconds
        self.ticks_left = None
        self.scoreline = None
        self.state = None
        if not skip_warning:
            print("WARNING: The Scoreboard object overwrites the inverted ball ang.vel. to include scoreboard, "
                  "make sure you're not using that and instead inverting on your own. "
                  "Call it in your obs builder's pre-step method to use.")

    def reset(self, initial_state: GameState):
        self.state = initial_state
        players_per_team = len(initial_state.players) // 2
        if self.random_resets:
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
            self.scoreline = b, o
        else:
            self.scoreline = 0, 0
            self.ticks_left = self.max_time_seconds * TICKS_PER_SECOND
        self.modify_gamestate(initial_state)

    def step(self, state: GameState, update_scores=True):
        if state != self.state:
            if state.ball.position[1] != 0:  # Don't count during kickoffs
                self.ticks_left = max(0, self.ticks_left - self.tick_skip)

            if update_scores:
                b, o = self.scoreline
                changed = False
                if state.blue_score > self.state.blue_score:  # Check in case of crash
                    b += state.blue_score - self.state.blue_score
                    changed = True
                if state.orange_score > self.state.orange_score:
                    o += state.orange_score - self.state.orange_score
                    changed = True
                tied = b == o
                if self.is_overtime():
                    if not tied:
                        self.ticks_left = float("-inf")  # Finished
                if self.ticks_left <= 0 and (state.ball.position[2] <= 110 or changed):
                    if tied:
                        self.ticks_left = float("inf")  # Overtime
                    else:
                        self.ticks_left = float("-inf")  # Finished
                self.scoreline = b, o

            self.state = state
        self.modify_gamestate(state)

    def modify_gamestate(self, state: GameState):
        state.inverted_ball.angular_velocity[:] = *self.scoreline, self.ticks_left

    def is_overtime(self):
        return self.ticks_left > 0 and math.isinf(self.ticks_left)

    def is_finished(self):
        return self.ticks_left < 0 and math.isinf(self.ticks_left)

    def win_prob(self):
        return win_prob(self.state.players // 2,
                        self.ticks_left * TICKS_PER_SECOND,
                        self.scoreline[0] - self.scoreline[1]).item()


FLOOR_AREA = 4 * BACK_WALL_Y * SIDE_WALL_X - 1152 * 1152  # Subtract corners
GOAL_AREA = GOAL_HEIGHT * 880


def win_prob(players_per_team, time_left_seconds, differential):
    # Utility function, calculates probability of blue team winning the full game
    from scipy.stats import skellam

    players_per_team = np.asarray(players_per_team)
    time_left_seconds = np.asarray(time_left_seconds)
    differential = np.asarray(differential)

    goal_floor_ratio = GOAL_AREA / (2 * GOAL_AREA + FLOOR_AREA)

    # inverted = np.random.random(differential.shape) > 0.5
    # differential[inverted] *= -1
    p = np.zeros(differential.shape)

    gpm = np.array(GOALS_PER_MIN)[players_per_team - 1]
    zero_seconds = (time_left_seconds <= 0) | np.isinf(time_left_seconds)

    mu_left = gpm * time_left_seconds / SECONDS_PER_MINUTE
    mu_left = mu_left[~zero_seconds]
    dist_left = skellam(mu_left, mu_left)

    diff_regulation = differential[~zero_seconds]

    # Probability of leading by two or more goals at end of regulation
    p[zero_seconds & (differential >= 2)] += 1
    p[~zero_seconds] += dist_left.cdf(diff_regulation - 2)

    # Probability of being tied at zero seconds and winning in overtime
    w = 0.5
    p[zero_seconds & (differential == 0)] += w
    p[~zero_seconds] += dist_left.pmf(diff_regulation) * w

    # Probability of leading by one at zero seconds, and surviving or getting scored on and winning OT
    w = (1 - goal_floor_ratio * 0.5)
    p[zero_seconds & (differential == 1)] += w
    p[~zero_seconds] += dist_left.pmf(diff_regulation - 1) * w

    # Probability of losing by one at zero seconds, and scoring and winning overtime
    w = goal_floor_ratio * 0.5
    p[zero_seconds & (differential == -1)] += w
    p[~zero_seconds] += dist_left.pmf(diff_regulation + 1) * w

    # p[inverted] = 1 - p[inverted]

    return p
