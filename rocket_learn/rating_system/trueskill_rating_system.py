import numpy as np
from trueskill import Rating, TrueSkill

from rocket_learn.rating_system.rating_system import RatingSystem


class TrueSkillRatingSystem(RatingSystem):
    def __init__(self, versions=None, env=None):
        super().__init__(versions)
        if env is None:
            env = TrueSkill()
        self.env = env
        self.ratings = {v: self.env.create_rating() for v in self.versions}

    def add_rating(self, identifier):
        latest = self.ratings[self.versions[-1]]
        self.versions[identifier] = len(self.versions)
        self.ratings[identifier] = self.env.create_rating(mu=latest.mu)

    def update(self, v1, v2, result):
        if v1 not in self.versions:
            self.add_rating(v1)
        if v2 not in self.versions:
            self.add_rating(v2)
        r1 = self.ratings[v1]
        r2 = self.ratings[v2]
        new_ratings = self.env.rate([{v1: r1}, {v2: r2}], [0, result])
        self.ratings.update(new_ratings)

    def predicted_win_rate(self, v1, v2):
        r1 = self.ratings[v1]
        r2 = self.ratings[v2]
        return win_probability([r1], [r2], env=self.env)


def win_probability(team1_ratings, team2_ratings, env=None):
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
