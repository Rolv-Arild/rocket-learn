from typing import Iterator, Optional

import numpy as np
from redis import Redis

from rocket_learn.rating_system.rating_system import RatingSystem
from rocket_learn.redis.utils import RedisKeys
from rocket_learn.rollout_generator.base_rollout_generator import BaseRolloutGenerator
from rocket_learn.utils.experience_buffer import ExperienceBuffer


class RedisRolloutGenerator(BaseRolloutGenerator):
    def __init__(self, redis: Redis,
                 max_age: int = 0,
                 rating_system: Optional[RatingSystem] = None,
                 matchup_queue_size: int = 100,
                 evaluation_probability=0.01,
                 past_version_probability=0.2):
        self.redis = redis
        self.max_age = max_age

        if rating_system is None:
            from rocket_learn.rating_system.trueskill_rating_system import TrueSkillRatingSystem
            rating_system = TrueSkillRatingSystem()
        self.rating_system = rating_system
        self.matchup_queue_size = matchup_queue_size

        self.evaluation_probability = evaluation_probability
        self.past_version_probability = past_version_probability

        self.latest_version = None

    def _fill_matchup_queue(self):
        size = self.redis.llen(RedisKeys.MATCHUPS)
        remaining_matchups = self.matchup_queue_size - size
        matchups = []
        # TODO deal with different gamemodes
        for _ in range(remaining_matchups):
            if np.random.random() < self.evaluation_probability:
                matchup = self.rating_system.select_matchup()
            elif np.random.random() < self.past_version_probability:
                pass  # TODO
            else:
                matchup = [self.latest_version]
            matchups.append(matchup)
        self.redis.rpush(RedisKeys.MATCHUPS, *matchups)

    def generate_rollouts(self) -> Iterator[ExperienceBuffer]:
        while True:
            self._fill_matchup_queue()
            data = self.redis.blpop(RedisKeys.ROLLOUTS)

    def update_parameters(self, new_params):
        pass
