from typing import Any

import numpy as np
import torch
import torch.distributions
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.obs_builders import AdvancedObs
from tabulate import tabulate
from torch import nn


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


# TODO use to print versions
def make_table(versions, ratings, blue, orange, pretrained_choice):
    version_info = []
    for v, r in zip(versions, ratings):
        if pretrained_choice is not None and v == 'na':  # print name but don't send it back
            version_info.append([str(type(pretrained_choice).__name__), "N/A"])
        elif v == 'na':
            version_info.append(['Human', "N/A"])
        else:
            if isinstance(v, int) and v < 0:
                v = f"Latest ({-v})"
            version_info.append([v, f"{r.mu:.2f}±{2 * r.sigma:.2f}"])

    blue_versions, blue_ratings = list(zip(*version_info[:blue]))
    orange_versions, orange_ratings = list(zip(*version_info[blue:]))

    if blue < orange:
        blue_versions += [""] * (orange - blue)
        blue_ratings += [""] * (orange - blue)
    elif orange < blue:
        orange_versions += [""] * (blue - orange)
        orange_ratings += [""] * (blue - orange)

    table_str = tabulate(list(zip(blue_versions, blue_ratings, orange_versions, orange_ratings)),
                         headers=["Blue", "rating", "Orange", "rating"], tablefmt="rounded_outline")

    return table_str
