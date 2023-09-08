from rocket_learn.rating_system.rating_system import RatingSystem


class EloRatingSystem(RatingSystem):
    def __init__(self, versions=None, initial=1000, K=32):
        super().__init__(versions)
        self.ratings = {v: initial for v in self.versions}
        self.initial = initial
        self.K = K

    def add_rating(self, identifier):
        self.versions[identifier] = len(self.versions)
        self.ratings[identifier] = self.initial

    def update(self, v1, v2, result):
        if v1 not in self.versions:
            self.add_rating(v1)
        if v2 not in self.versions:
            self.add_rating(v2)
        result = 1 if result > 0 else 0 if result < 0 else 0.5
        expected = self.predicted_win_rate(v1, v2)
        update = self.K * (result - expected)
        self.ratings[v1] += update
        self.ratings[v2] -= update

    def predicted_win_rate(self, v1, v2):
        r1 = self.ratings.get(v1, self.initial)
        r2 = self.ratings.get(v2, self.initial)
        return 1 / (1 + 10 ** ((r2 - r1) / 400))


