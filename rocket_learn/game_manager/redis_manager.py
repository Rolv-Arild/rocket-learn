import functools
from typing import Literal, Tuple

import numpy as np
from redis import Redis

from rocket_learn.agent.agent import Agent
from rocket_learn.agent.policy import Policy
from rocket_learn.envs.rocket_league import RocketLeague
from rocket_learn.game_manager.default_manager import DefaultManager, DefaultMatchup
from rocket_learn.rollout_generator.redis.utils import VERSION_LATEST, MODEL_LATEST, OPPONENT_MODELS, \
    _unserialize_model, get_rating, LATEST_RATING_ID
from rocket_learn.utils.util import probability_NvsM


class RedisManager(DefaultManager):
    def __init__(self,
                 env: RocketLeague,
                 redis: Redis,
                 past_model_prob: float,
                 eval_prob: float,
                 gamemode_weights: dict[str, float],
                 display: Literal[None, "stochastic", "deterministic", "rollout"] = None,
                 local_cache=None,
                 full_team_rollouts=False,
                 full_team_evals=True,
                 target_sigma=1):
        super().__init__(env, gamemode_weights, display)
        self.redis = redis

        self.past_model_prob = past_model_prob
        self.eval_prob = eval_prob

        self.full_team_rollouts = full_team_rollouts
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

    def _determine_gamemode(self):
        modes = list(self.gamemode_weights.keys())
        target_dist = np.array(list(self.gamemode_weights.values()))
        target_dist = target_dist / target_dist.sum()

        current_dist = np.array([self.gamemode_exp[k] for k in modes])

        assert (target_dist > 0).all()

        diff = current_dist - target_dist
        # Index will be random between ties of lowest diff
        idx = np.random.choice(np.where(diff == diff.min())[0])

        mode = modes[idx]

        # In theory 1v2 == 2v1, we let user decide on one as a name then we swap randomly
        b, o = mode.split("v")
        b = int(b)
        o = int(o)
        if o != b and np.random.random() > 0.5:
            b, o = o, b

        return mode, b, o

    @functools.lru_cache(maxsize=8)
    def _get_agent(self, identifier: str):
        # TODO update to return an Agent object instead of a Policy
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

    def _get_matchup_latest(self, deterministic):
        latest_model = _unserialize_model(self.redis.get(MODEL_LATEST))
        latest_model.deterministic = deterministic

        mode, b, o = self._determine_gamemode()

        cars_models = {f"{color}-{n}": latest_model
                       for color in ("blue", "orange")
                       for n in range(b if color == 'blue' else o)}

        return cars_models

    def _get_matchup_eval(self):  # TODO somehow push the result of eval matches to redis
        # Assume equal gamemode importance, maybe select based on sigma in future?
        gamemode = np.random.choice(self.gamemode_weights.keys())
        all_ratings = get_rating(gamemode, None, self.redis)

        identifiers, ratings = zip(*all_ratings.items())

        # Select target rating, aim for high sigma
        probs = np.array([r.sigma for r in ratings])
        probs = np.clip(probs - self.target_sigma, 0)
        s = probs.sum()
        if s <= 0:  # TODO let users pick a different probability if no models have low sigma?
            probs[:] = 1
            s = len(probs)
        probs /= s

        target_idx = np.random.choice(len(ratings), p=probs)
        target_identifier = identifiers[target_idx]
        target_rating = ratings[target_idx]

        # Imbalanced modes are not entirely thought through but might not crash at least
        b, o = (int(v) for v in gamemode.split("v"))
        if b != o and np.random.random() < 0.5:
            b, o = o, b

        # Weight opponents by likelihood of winning one game each in two games
        probs = np.zeros(len(ratings))
        for idx, rating in enumerate(ratings):
            if idx != target_idx:  # Keep target at 0, no matchups against itself
                p = probability_NvsM(b * [target_rating], o * [rating])
                probs[idx] = p * (1 - p)
        s = probs.sum()
        if s <= 0:
            return None
        probs /= s

        if self.full_team_evals:
            # Select a single opponent
            if len(ratings) < 2:
                return None
            opponent_idx = np.random.choice(len(ratings), p=probs)
            opponent_identifier = identifiers[opponent_idx]
            opponent_rating = ratings[opponent_idx]

            target_agent: Agent = self._get_agent(target_identifier)
            opponent_agent: Agent = self._get_agent(opponent_identifier)

            identifier_agent = {target_identifier: target_agent,
                                opponent_identifier: opponent_agent}
            identifier_rating = {target_identifier: target_rating,
                                 opponent_identifier: opponent_rating}

            ids = [target_identifier, opponent_identifier]
            np.random.shuffle(ids)

            car_identifier = {f"{color}-{n}": identifier
                              for color, identifier in (("blue", ids[0]), ("orange", ids[1]))
                              for n in range(b if color == 'blue' else o)}
        else:
            # Select several random opponents
            if len(ratings) < b + o:
                return None
            opponent_idx = np.random.choice(len(ratings), p=probs, size=b + o - 1, replace=False).tolist()

            cars = [f"{color}-{n}"
                    for color in ("blue", "orange")
                    for n in range(b if color == 'blue' else o)]

            opponent_idx.append(target_idx)
            np.random.shuffle(opponent_idx)

            car_identifier = {}
            identifier_agent = {}
            identifier_rating = {}
            for car, idx in zip(cars, opponent_idx):
                identifier = identifiers[idx]
                rating = ratings[idx]

                car_identifier[car] = identifier
                if identifier not in identifier_agent:
                    identifier_agent[identifier] = self._get_agent(identifier)
                    identifier_rating[identifier] = rating

        return car_identifier, identifier_agent, identifier_rating

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

        car_identifier = {}
        identifier_agent = {latest_identifier: latest_agent}
        identifier_rating = {latest_identifier: latest_rating}

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
        idx = np.random.permutation(len(selected_identifiers))

        cars = [f"{color}-{n}"
                for color in ("blue", "orange")
                for n in range(b if color == 'blue' else o)]

        for car, i in zip(cars, idx):
            identifier = selected_identifiers[i]
            rating = selected_ratings[i]

            car_identifier[car] = identifier
            if identifier not in identifier_agent:
                identifier_agent[identifier] = self._get_agent(identifier)
                identifier_rating[identifier] = rating

        return car_identifier, identifier_agent, identifier_rating

    def generate_matchup(self) -> Tuple[DefaultMatchup, int]:
        if self.eval_prob < np.random.random():
            car_identifier, identifier_agent, identifier_rating = \
                self._get_matchup_eval()
            matchup_type = self.EVAL
        elif self.display in ("stochastic", "deterministic") \
                or self.past_model_prob < np.random.random():
            car_identifier, identifier_agent, identifier_rating = \
                self._get_matchup_latest(self.display == "deterministic")
            matchup_type = self.SHOW if self.display else self.ROLLOUT
        else:
            car_identifier, identifier_agent, identifier_rating \
                = self._get_matchup_rollout()
            matchup_type = self.SHOW if self.display else self.ROLLOUT

        # TODO print matchup with ratings? Or should worker be in charge of that?

        return (car_identifier, identifier_agent), matchup_type
