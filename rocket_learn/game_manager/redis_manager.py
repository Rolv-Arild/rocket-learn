import functools
from typing import Dict, Literal

import numpy as np
from pettingzoo import ParallelEnv
from redis import Redis
from trueskill import quality

from rocket_learn.agent.agent import Agent
from rocket_learn.agent.policy import Policy
from rocket_learn.game_manager.default_manager import DefaultManager
from rocket_learn.rollout_generator.redis.utils import VERSION_LATEST, MODEL_LATEST, OPPONENT_MODELS, \
    _unserialize_model, get_rating, LATEST_RATING_ID
from rocket_learn.utils.util import probability_NvsM


class RedisManager(DefaultManager):
    def __init__(self, env: ParallelEnv, redis: Redis, past_model_prob: float, eval_prob: float,
                 gamemode_weights, local_cache=None, full_team_evals=True, target_sigma=1):
        super().__init__(env)
        self.redis = redis

        self.past_model_prob = past_model_prob
        self.eval_prob = eval_prob

        self.full_team_evals = full_team_evals
        self.target_sigma = target_sigma

        self.sql = None
        if local_cache:
            import sqlite3 as sql
            self.sql = sql.connect('redis-model-cache-' + local_cache + '.db')
            # if the table doesn't exist in the database, make it
            self.sql.execute("""
                CREATE TABLE if not exists MODELS (
                    id TEXT PRIMARY KEY,
                    parameters BLOB NOT NULL
                );
            """)

    @functools.lru_cache(maxsize=8)
    def _get_agent(self, identifier: str):
        if self.sql is not None:
            models = self.sql.execute("SELECT parameters FROM MODELS WHERE id == ?", (identifier,)).fetchall()
            if len(models) == 0:
                bytestream = self.redis.hget(OPPONENT_MODELS, identifier)
                model = _unserialize_model(bytestream)

                self.sql.execute('INSERT INTO MODELS (id, parameters) VALUES (?, ?)', (identifier, bytestream))
                self.sql.commit()
            else:
                # should only ever be 1 version of parameters
                assert len(models) <= 1
                # stored as tuple due to sqlite,
                assert len(models[0]) == 1

                bytestream = models[0][0]
                model = _unserialize_model(bytestream)
        else:
            model = _unserialize_model(self.redis.hget(OPPONENT_MODELS, identifier))

        if identifier.endswith("deterministic") and isinstance(model, Policy):
            model.deterministic = True

        return model

    def _determine_gamemode(self):
        modes = list(self.gamemode_weights.keys())
        target_dist = np.array(list(self.gamemode_weights.values()))
        target_dist = target_dist / target_dist.sum()

        current_dist = np.array([self.gamemode_exp[k] for k in modes])

        assert (target_dist > 0).all()

        diff = current_dist - target_dist
        idx = np.random.choice(np.where(diff == diff.min())[0])

        mode = modes[idx]

        b, o = mode.split("v")
        b = int(b)
        o = int(o)
        if o != b and np.random.random() > 0.5:
            b, o = o, b

        return mode, b, o

    def _get_matchup_latest(self, deterministic):
        latest_model = _unserialize_model(self.redis.get(MODEL_LATEST))
        latest_model.deterministic = deterministic

        mode, b, o = self._determine_gamemode()

        cars_models = {f"{color}-{n}": latest_model
                       for color in ("blue", "orange")
                       for n in range(b if color == 'blue' else o)}

        return cars_models

    def _get_matchup_eval(self):
        gamemode = np.random.choice(self.gamemode_weights.keys())
        all_ratings = get_rating(gamemode, None, self.redis)

        identifiers, ratings = zip(*all_ratings.items())
        probs = np.array([r.sigma for r in ratings])
        probs = np.clip(probs - self.target_sigma, 0)
        s = probs.sum()
        if s <= 0:
            probs[:] = 1
            s = len(probs)
        probs /= s

        target_idx = np.random.choice(len(ratings), p=probs)
        target_identifier = identifiers[target_idx]
        target_rating = ratings[target_idx]

        b, o = (int(v) for v in gamemode.split("v"))
        if b != o and np.random.random() < 0.5:
            b, o = o, b

        probs = np.zeros(len(ratings))
        for idx, rating in enumerate(ratings):
            if idx != target_idx:
                p = probability_NvsM(b * [target_rating], o * [rating])
                probs[idx] = p * (1 - p)
        s = probs.sum()
        if s <= 0:
            return None
        probs /= s

        if self.full_team_evals:
            opponent_idx = np.random.choice(len(ratings), p=probs)
            opponent_identifier = identifiers[opponent_idx]
            opponent_rating = ratings[opponent_idx]

            a1: Agent = self._get_agent(target_identifier)
            a2: Agent = self._get_agent(opponent_identifier)

            a1.rating = target_rating
            a2.rating = opponent_rating

            a1.identifier = target_identifier
            a2.identifier = opponent_identifier

            if b == o and np.random.random() < 0.5:
                a1, a2 = a2, a1

            cars_models = {f"{color}-{n}": model
                           for color, model in (("blue", a1), ("orange", a2))
                           for n in range(b if color == 'blue' else o)}
            return cars_models
        else:
            opponent_idx = np.random.choice(len(ratings), p=probs, size=b + o - 1).tolist()

            cars = [f"{color}-{n}"
                    for color in ("blue", "orange")
                    for n in range(b if color == 'blue' else o)]

            opponent_idx.append(target_idx)
            np.random.shuffle(opponent_idx)

            cars_models = {}
            for car, idx in zip(cars, opponent_idx):
                identifier = identifiers[idx]
                rating = ratings[idx]
                agent = self._get_agent(identifier)
                agent.identifier = identifier
                agent.rating = rating
                cars_models[car] = agent

            return cars_models

    def _get_matchup_rollout(self):
        mode, b, o = self._determine_gamemode()
        all_ratings = get_rating(mode, None, self.redis)

        identifiers, ratings = zip(*all_ratings.items())

        n_old = np.random.randint(1, o + b)
        n_new = o + b - n_old

        latest_identifier = self.redis.get(LATEST_RATING_ID)
        latest_rating = all_ratings[latest_identifier]

        latest_agent = self.redis.get(MODEL_LATEST)
        latest_identifier = self.redis.get(VERSION_LATEST)

        probs = np.zeros(len(ratings))
        for idx, rating in enumerate(ratings):
            p = probability_NvsM(b * [latest_rating], o * [rating])
            probs[idx] = p * (1 - p)
        s = probs.sum()
        if s <= 0:
            return None
        probs /= s

        idx = np.random.choice(len(probs), p=probs, size=n_old)

        selected_identifiers = n_new * [latest_identifier] + [identifiers[i] for i in idx]
        selected_ratings = n_new * [latest_rating] + [ratings[i] for i in idx]

        # Shuffle
        idx = np.random.choice(len(selected_identifiers), size=len(selected_identifiers), replace=False)

        cars = [f"{color}-{n}"
                for color in ("blue", "orange")
                for n in range(b if color == 'blue' else o)]

        cars_models = {}
        for car, i in zip(cars, idx):
            identifier = selected_identifiers[i]
            rating = selected_ratings[i]
            if identifier == latest_identifier:
                agent = latest_agent
            else:
                agent = self._get_agent(identifier)
            agent.identifier = identifier
            agent.rating = rating
            cars_models[car] = agent

        return cars_models

    def generate_matchup(self) -> (Dict[str, Agent], int):
        if self.eval_prob < np.random.random():
            cars_models = self._get_matchup_eval()
            return cars_models, self.EVAL
        elif self.display in ("stochastic", "deterministic") \
                or self.past_model_prob < np.random.random():
            cars_models = self._get_matchup_latest(self.display == "deterministic")
            return cars_models, self.SHOW if self.display else self.ROLLOUT
        else:
            cars_models = self._get_matchup_rollout()
            return cars_models, self.SHOW if self.display else self.ROLLOUT
