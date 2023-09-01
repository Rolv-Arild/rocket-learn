import numpy as np
from rlgym.rocket_league.common_values import BACK_WALL_Y, GOAL_HEIGHT, SIDE_WALL_X

TICKS_PER_SECOND = 120
SECONDS_PER_MINUTE = 60

RAMP_RADIUS = 256
CORNER_CATHETUS_LENGTH = 1152
CORNER_AREA = (CORNER_CATHETUS_LENGTH + RAMP_RADIUS) ** 2 / 2
FLOOR_AREA = 4 * ((BACK_WALL_Y - RAMP_RADIUS) * (SIDE_WALL_X - RAMP_RADIUS) - CORNER_AREA)
GOAL_AREA = GOAL_HEIGHT * 880

DEFAULT_GOALS_PER_MIN = {(1, 1): 0.845, (2, 2): 0.535, (3, 3): 0.415}  # Stats from GC+ replays


def win_prob(gpm, time_left_seconds, differential):
    """
    Utility function, calculates probability of team winning the full game
    """
    from scipy.stats import skellam  # Optional dependency

    time_left_seconds = np.asarray(time_left_seconds)
    differential = np.asarray(differential)

    # Assume each unit of area that ends the episode is equally likely
    # This gives about 0.8% chance, RLCSX stats say 1.7% but this will change massively based on skill and gamemode
    goal_floor_ratio = GOAL_AREA / (2 * GOAL_AREA + FLOOR_AREA)

    # inverted = np.random.random(differential.shape) > 0.5
    # differential[inverted] *= -1
    p = np.zeros(differential.shape)

    zero_seconds = (time_left_seconds <= 0) | np.isinf(time_left_seconds)

    if isinstance(gpm, tuple):  # Assume different gpm per team
        mu_left_0, mu_left_1 = gpm
        mu_left_0 = mu_left_0 * time_left_seconds / SECONDS_PER_MINUTE
        mu_left_1 = mu_left_1 * time_left_seconds / SECONDS_PER_MINUTE
        mask = ~zero_seconds
        mu_left_0 = mu_left_0[mask]
        mu_left_1 = mu_left_1[mask]
        dist_left = skellam(mu_left_0, mu_left_1)
    else:  # Assume same gpm per team
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
