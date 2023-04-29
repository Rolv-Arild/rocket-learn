import numpy as np

from rlgym.utils.common_values import BACK_WALL_Y, SIDE_WALL_X, GOAL_HEIGHT
from scipy.stats import skellam

TICKS_PER_SECOND = 120
SECONDS_PER_MINUTE = 60
GOALS_PER_MIN = (1, 0.6, 0.45)  # Stats from ballchasing, S14 GC (before SSL)

FLOOR_AREA = 4 * BACK_WALL_Y * SIDE_WALL_X - 1152 * 1152  # Subtract corners
GOAL_AREA = GOAL_HEIGHT * 880


def win_prob(players_per_team, time_left_seconds, differential):
    # Utility function, calculates probability of blue team winning the full game

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
