from typing import Any

import numpy as np
import torch
import torch.distributions
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.obs_builders import AdvancedObs
from torch import nn


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class SplitLayer(nn.Module):
    def __init__(self, splits=None):
        super().__init__()
        if splits is not None:
            self.splits = splits
        else:
            self.splits = (3,) * 5 + (2,) * 3

    def forward(self, x):
        return torch.split(x, self.splits, dim=-1)


# TODO AdvancedObs should be supported by default, use stack instead of cat
class ExpandAdvancedObs(AdvancedObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        return np.reshape(
            super(ExpandAdvancedObs, self).build_obs(player, state, previous_action),
            (1, -1)
        )


def probability_NvsM(team1_ratings, team2_ratings, env=None):
    from trueskill import global_env
    # Trueskill extension, source: https://github.com/sublee/trueskill/pull/17
    """Calculates the win probability of the first team over the second team.
    :param team1_ratings: ratings of the first team participants.
    :param team2_ratings: ratings of another team participants.
    :param env: the :class:`TrueSkill` object.  Defaults to the global
                environment.
    """
    if env is None:
        env = global_env()

    team1_mu = sum(r.mu for r in team1_ratings)
    team1_sigma = sum((env.beta ** 2 + r.sigma ** 2) for r in team1_ratings)
    team2_mu = sum(r.mu for r in team2_ratings)
    team2_sigma = sum((env.beta ** 2 + r.sigma ** 2) for r in team2_ratings)

    x = (team1_mu - team2_mu) / np.sqrt(team1_sigma + team2_sigma)
    probability_win_team1 = env.cdf(x)
    return probability_win_team1
