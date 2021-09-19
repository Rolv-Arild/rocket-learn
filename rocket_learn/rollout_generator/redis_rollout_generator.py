import os
import pickle
import time
from typing import Iterator
from uuid import uuid4

import msgpack
import msgpack_numpy as m
import numpy as np
import matplotlib.pyplot  # noqa
import wandb
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from redis import Redis
from redis.exceptions import ResponseError
from trueskill import Rating, rate

from rlgym.envs import Match
from rlgym.gamelaunch import LaunchPreference
from rlgym.gym import Gym
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.rollout_generator.base_rollout_generator import BaseRolloutGenerator
from rocket_learn.utils import util
from rocket_learn.utils.util import softmax

# Constants for consistent key lookup
QUALITIES = "qualities"
N_UPDATES = "num-updates"
SAVE_FREQ = "save-freq"

MODEL_LATEST = "model-latest"
VERSION_LATEST = "model-version"
QUALITY_LATEST = "model-rating"

ROLLOUTS = "rollout"
OPPONENT_MODELS = "opponent-models"
WORKER_IDS = "worker-ids"
_ALL = (
    QUALITIES, N_UPDATES, SAVE_FREQ, MODEL_LATEST, ROLLOUTS, VERSION_LATEST, QUALITY_LATEST, OPPONENT_MODELS,
    WORKER_IDS)

m.patch()


# Helper methods for easier changing of byte conversion
def _serialize(obj):
    return msgpack.packb(obj)


def _unserialize(obj):
    return msgpack.unpackb(obj)


def _serialize_model(mdl):
    return pickle.dumps(mdl)
    # buf = io.BytesIO()
    # torch.save([mdl.actor, mdl.critic, mdl.shared], buf)
    # return buf


def _unserialize_model(buf):
    agent = pickle.loads(buf)
    return agent.cpu()
    # return torch.load(buf)


class RedisRolloutGenerator(BaseRolloutGenerator):
    """
    Rollout generator in charge of sending commands to workers via redis
    """

    def __init__(self, redis: Redis, save_every=10, logger=None, clear=True):
        # **DEFAULT NEEDS TO INCORPORATE BASIC SECURITY, THIS IS NOT SUFFICIENT**
        self.redis = redis
        self.logger = logger

        # TODO saving/loading
        if clear:
            for key in _ALL:
                if self.redis.exists(key) > 0:
                    self.redis.delete(key)
            self.redis.set(N_UPDATES, 0)
            self.redis.set(SAVE_FREQ, save_every)
            self.redis.set(QUALITY_LATEST, _serialize((0, 1)))

    def generate_rollouts(self) -> Iterator[ExperienceBuffer]:
        while True:
            rollout_bytes = self.redis.blpop(ROLLOUTS)[1]

            rollout_data, uuid, name, result = _unserialize(rollout_bytes)
            latest_version = int(self.redis.get(VERSION_LATEST))

            # TODO log uuid and name

            blue_players = sum(divmod(len(rollout_data), 2))
            blue, orange = [], []
            rollouts = []
            for n, (rollout, version) in enumerate(rollout_data):
                if version < 0:
                    if version == latest_version:
                        rollout = ExperienceBuffer(None, *rollout)
                        rollouts.append(rollout)
                        rating = Rating(*_unserialize(self.redis.get(QUALITY_LATEST)))
                    else:
                        break
                else:
                    rating = Rating(*_unserialize(self.redis.lindex(QUALITIES, version)))

                if n < blue_players:
                    blue.append(rating)
                else:
                    orange.append(rating)
            else:  # Latest version is not outdated
                if result >= 0:
                    r1, r2 = rate((blue, orange), ranks=(0, result))
                else:
                    r2, r1 = rate((orange, blue))

                versions = {}
                for rating, (rollout, version) in zip(r1 + r2, rollout_data):
                    versions.setdefault(version, []).append(rating)

                for version, ratings in versions.items():
                    avg_rating = Rating((sum(r.mu for r in ratings) / len(ratings)),
                                        (sum(r.sigma for r in ratings) / len(ratings)))
                    if version > 0:  # Old
                        self.redis.lset(QUALITIES, version, _serialize(tuple(avg_rating)))
                    elif abs(version - latest_version) <= 1:
                        self.redis.set(QUALITY_LATEST, _serialize(tuple(avg_rating)))

                yield from rollouts

    def _add_opponent(self, agent):
        # Add to list
        self.redis.rpush(OPPONENT_MODELS, agent)
        # Set quality
        ratings = [Rating(*_unserialize(v)) for v in self.redis.lrange(QUALITIES, 0, -1)]
        if ratings:
            mus = np.array([r.mu for r in ratings])
            conf = np.array([r.sigma for r in ratings])
            fig = Figure()
            ax: Axes = fig.subplots()
            ax.plot(np.arange(len(ratings)), mus, color="royalblue")
            ax.fill_between(np.arange(len(ratings)), mus + 3 * conf, mus - 3 * conf, color="lightsteelblue")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("TrueSkill")
            ax.set_title("Qualities")

            self.logger.log({
                "qualities": wandb.Image(fig),
                # "qualities": wandb.plot.line_series(np.arange(len(mus)), [mus],
                #                                     # noqa
                #                                     ["quality"], "Qualities", "version")
            }, commit=False)
            quality = Rating(ratings[-1].mu)  # Same mu, reset sigma
        else:
            quality = Rating(0, 1)  # First agent is fixed at 0
        self.redis.rpush(QUALITIES, _serialize(tuple(quality)))

    def update_parameters(self, new_params):
        """
        update redis (and thus workers) with new model data and save data as future opponent
        :param new_params: new model parameters
        """
        model_bytes = _serialize_model(new_params)
        self.redis.set(MODEL_LATEST, model_bytes)
        self.redis.decr(VERSION_LATEST)
        # Same mu, reset sigma
        self.redis.set(QUALITY_LATEST, _serialize(tuple(Rating(_unserialize(self.redis.get(QUALITY_LATEST))[0]))))

        # TODO Idea: workers send name to identify who contributed rollouts,
        # keep track of top rollout contributors (each param update and total)
        # Also UID to keep track of current number of contributing workers?

        n_updates = self.redis.incr(N_UPDATES) - 1
        save_freq = int(self.redis.get(SAVE_FREQ))

        if n_updates % save_freq == 0:
            # self.redis.set(MODEL_N.format(self.n_updates // self.save_every), model_bytes)
            self._add_opponent(model_bytes)
            try:
                self.redis.save()
            except ResponseError:
                print("redis manual save aborted, save already in progress")


class RedisRolloutWorker:
    """
    Provides RedisRolloutGenerator with rollouts via a Redis server
    """

    def __init__(self, redis: Redis, name: str, match: Match, current_version_prob=.9):
        # TODO model or config+params so workers can recreate just from redis connection?
        self.redis = redis
        self.name = name

        self.current_agent = _unserialize_model(self.redis.get(MODEL_LATEST))
        self.current_version_prob = current_version_prob

        # **DEFAULT NEEDS TO INCORPORATE BASIC SECURITY, THIS IS NOT SUFFICIENT**
        self.uuid = str(uuid4())
        self.redis.rpush(WORKER_IDS, self.uuid)
        print("Started worker", self.uuid, "on host", self.redis.connection_pool.connection_kwargs.get("host"),
              "under name", name)  # TODO log instead
        self.match = match
        self.env = Gym(match=self.match, pipe_id=os.getpid(), launch_preference=LaunchPreference.EPIC_LOGIN_TRICK,
                       use_injector=True)
        self.n_agents = self.match.agents

    def _get_opponent_index(self):
        # Get qualities
        # TODO do multiple selections at once
        # sum priorities to have higher chance of selecting low sigma
        qualities = np.asarray([sum(_unserialize(v)) for v in self.redis.lrange(QUALITIES, 0, -1)])
        # Pick opponent
        probs = softmax(qualities / np.log(10))
        index = np.random.choice(len(probs), p=probs)
        return index, probs[index]

    def run(self):  # Mimics Thread
        """
        begin processing in already launched match and push to redis
        """
        n = 0
        while True:
            model_bytes = self.redis.get(MODEL_LATEST)
            latest_version = self.redis.get(VERSION_LATEST)
            if model_bytes is None:
                time.sleep(1)
                continue  # Wait for model to get published
            updated_agent = _unserialize_model(model_bytes)
            latest_version = int(latest_version)

            n += 1

            self.current_agent = updated_agent

            # TODO customizable past agent selection, should team only be same agent?
            agents = [(self.current_agent, latest_version, self.current_version_prob)]  # Use at least one current agent

            if self.n_agents > 1:
                # Ensure final proportion is same
                adjusted_prob = (self.current_version_prob * self.n_agents - 1) / (self.n_agents - 1)
                for i in range(self.n_agents - 1):
                    is_current = np.random.random() < adjusted_prob
                    if not is_current:
                        index, prob = self._get_opponent_index()
                        version = index
                        selected_agent = _unserialize_model(self.redis.lindex(OPPONENT_MODELS, index))
                    else:
                        prob = self.current_version_prob
                        version = latest_version
                        selected_agent = self.current_agent

                    agents.append((selected_agent, version, prob))

            np.random.shuffle(agents)
            print("Generating rollout with versions:", [v for a, v, p in agents])

            rollouts, result = util.generate_episode(self.env, [agent for agent, version, prob in agents])

            rollout_data = []
            for rollout, (agent, version, prob) in zip(rollouts, agents):
                rollout_data.append((
                    (rollout.observations, rollout.actions, rollout.rewards, rollout.dones, rollout.log_prob),
                    version
                ))

            self.redis.rpush(ROLLOUTS, _serialize((rollout_data, self.uuid, self.name, result)))
