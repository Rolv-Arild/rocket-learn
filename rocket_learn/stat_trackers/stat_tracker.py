import abc
import warnings
from typing import List

import numpy as np
from rlgym.api import AgentID
from rlgym.rocket_league.api import GameState


def aggregate_ratio(items):
    num = 0
    den = 0
    for a, b in items:
        num += a
        den += b
    return num / (den or 1)


AGGREGATION_METHODS = {
    "agg_ratio": aggregate_ratio,
    "mean": np.mean,
    "std": np.std,
    "max": max,
    "min": min,
}


class StatTracker(abc.ABC):
    def __init__(self, name, aggregation_method):
        self.name = name
        if aggregation_method not in AGGREGATION_METHODS:
            warnings.warn(f"Warning: aggregation method '{aggregation_method}'"
                          f" is not present in the aggregation method dictionary")
        self.aggregation_method = aggregation_method

    def collect(self, agent_ids: List[AgentID], game_states: List[GameState]):
        raise NotImplementedError
