from pettingzoo import ParallelEnv
from redis import Redis

from rocket_learn.game_manager.game_manager import GameManager
from rocket_learn.rollout_generator.redis.utils import EVALUATIONS, _serialize


class RedisRolloutWorker:
    def __init__(self, redis: Redis, name: str, manager: GameManager, env: ParallelEnv):
        self.redis = redis
        self.name = name
        self.manager = manager
        self.env = env

    def run(self):
        while True:
            matchup, mu_type = self.manager.generate_matchup()
            if mu_type == GameManager.SHOW:
                self.manager.show(matchup)
            elif mu_type == GameManager.ROLLOUT:
                result = self.manager.evaluate(matchup)
                self.redis.rpush(EVALUATIONS, _serialize(({k: a.identifier for k, a in matchup.items()}, result)))
            elif mu_type == GameManager.EVAL:
                agents_rollouts = self.manager.show(matchup)
            else:
                raise ValueError
