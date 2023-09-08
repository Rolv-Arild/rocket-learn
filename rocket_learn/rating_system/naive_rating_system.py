import random

import numpy as np

from rocket_learn.rating_system.rating_system import RatingSystem


class NaiveRatingSystem(RatingSystem):
    def __init__(self, versions=None):
        super().__init__(versions)
        self.wins = {(v1, v2): 0 for v1 in self.versions for v2 in self.versions}
        self.losses = {(v1, v2): 0 for v1 in self.versions for v2 in self.versions}

    def add_rating(self, identifier):
        self.versions[identifier] = len(self.versions)
        self.wins.update({(identifier, v): 0 for v in self.versions})
        self.losses.update({(identifier, v): 0 for v in self.versions})
        self.wins.update({(v, identifier): 0 for v in self.versions})
        self.losses.update({(v, identifier): 0 for v in self.versions})

    def update(self, v1, v2, result):
        if v1 not in self.versions:
            self.add_rating(v1)
        if v2 not in self.versions:
            self.add_rating(v2)
        if result > 0:
            self.wins[(v1, v2)] += 1
            self.losses[(v2, v1)] += 1
        elif result < 0:
            self.losses[(v1, v2)] += 1
            self.wins[(v2, v1)] += 1

    def predicted_win_rate(self, v1, v2):
        wins = self.wins.get((v1, v2), 0)
        losses = self.losses.get((v1, v2), 0)
        total = wins + losses
        if total == 0:
            return float("nan")
        return wins / total

    def get_beta_dist(self):
        from scipy.stats.distributions import beta
        record = np.array([[
            [self.losses[(k1, k2)], self.wins[(k1, k2)]]
            for k1 in sorted(self.versions)]
            for k2 in sorted(self.versions)]
        )
        dist = beta(record[..., 1] + 1, record[..., 0] + 1)
        return dist

    def select_matchup(self):
        dist = self.get_beta_dist()

        stdev = dist.std()
        m = stdev.max()
        potential_matchups = list(zip(*np.where(stdev == m)))

        return random.choice(potential_matchups)
