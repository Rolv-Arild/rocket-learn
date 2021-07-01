import time
from typing import Generator

from redis import Redis

from rocket_learn.rollout_generator.base_rollout_generator import BaseRolloutGenerator

# Hopefully we only need this one file, so this is where it belongs
QUALITIES = "qualities"
MODEL_LATEST = "model-latest"
MODEL_N = "model-{}"
ROLLOUTS = "rollout"


class RedisRolloutGenerator(BaseRolloutGenerator):
    def __init__(self, save_every=10):
        self.redis = Redis()
        self.n_updates = 0
        self.save_every = save_every

    def generate_rollouts(self) -> Generator:
        while True:
            rollout = self.redis.lpop(ROLLOUTS)
            if rollout is not None:  # Assuming nil is converted to None by py-redis
                yield rollout
            time.sleep(1)  # Don't DOS Redis

    def update_parameters(self, new_params):
        self.redis.set(MODEL_LATEST, new_params)
        if self.n_updates % self.save_every == 0:
            self.redis.set(MODEL_N.format(self.n_updates // self.save_every), new_params)


class RedisRolloutWorker:
    pass  # TODO move worker code in here
