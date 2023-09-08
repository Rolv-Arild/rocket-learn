import math

import numpy as np


class RatingSystem:
    def __init__(self, versions=None):
        self.versions: dict = {} if versions is None else versions

    def add_rating(self, identifier):
        raise NotImplementedError

    def update(self, v1, v2, result):
        raise NotImplementedError

    def all_predicted_win_rates(self):
        win_rates = {}
        for v1, i in self.versions.items():
            win_rates[(v1, v1)] = 0.5
            for v2, j in self.versions.items():
                if j <= i:
                    continue
                wr = self.predicted_win_rate(v1, v2)
                win_rates[(v1, v2)] = wr
                win_rates[(v2, v1)] = 1 - wr
        return win_rates

    def skills(self):
        win_rates = self.all_predicted_win_rates()
        skills = {}
        for v in self.versions:
            p = 0
            n = 0
            for v2 in self.versions:
                wr = win_rates[(v, v2)]
                if math.isfinite(wr):
                    p += wr
                    n += 1
            p /= n
            odds = p / (1 - p)
            skills[v] = math.log(odds)
        return skills

    def predicted_win_rate(self, v1, v2):
        raise NotImplementedError

    def select_matchup(self):
        win_rates = self.all_predicted_win_rates()

        matchups, win_probs = zip(*win_rates.items())
        scores = np.zeros(len(matchups))

        for i, (matchup, win_prob) in enumerate(zip(matchups, win_probs)):
            scores[i] = win_prob * (1 - win_prob)
        probs = scores / scores.sum()
        idx = np.random.choice(len(matchups), p=probs)
        return matchups[idx]

    def __str__(self):
        s = f"{self.__class__.__name__}("
        for key, value in self.__dict__.items():
            if key != "versions" and isinstance(value, (float, int, str)) and not callable(value):
                v = str(value)
                if len(v) <= 10:
                    s += f"{key}={value},"
        if s[-1] == ",":
            s = s[:-1]
        s += ")"
        return s
