import os
from threading import Thread

import cloudpickle as pickle
import time
from collections import Counter
from typing import Iterator, Callable
from uuid import uuid4

import msgpack
import msgpack_numpy as m
import numpy as np
# import matplotlib.pyplot  # noqa
import wandb
# from matplotlib.axes import Axes
# from matplotlib.figure import Figure
from redis import Redis
from redis.exceptions import ResponseError
from rlgym.utils import ObsBuilder, RewardFunction
from rlgym.utils.obs_builders import AdvancedObs
from trueskill import Rating, rate
import plotly.graph_objs as go

from rlgym.envs import Match
from rlgym.gamelaunch import LaunchPreference
from rlgym.gym import Gym
from rlgym.utils.gamestates import GameState
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.rollout_generator.base_rollout_generator import BaseRolloutGenerator
from rocket_learn.utils import util
from rocket_learn.utils.util import softmax, encode_gamestate

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

    def __init__(
            self,
            redis: Redis,
            obs_build_factory: Callable[[], ObsBuilder],
            rew_build_factory: Callable[[], RewardFunction],
            save_every=10,
            logger=None,
            clear=True
    ):
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

        self.contributors = Counter()  # No need to save, clears every iteration
        self.obs_build_factory = obs_build_factory
        self.rew_build_factory = rew_build_factory

    def generate_rollouts(self) -> Iterator[ExperienceBuffer]:
        while True:
            rollout_bytes = self.redis.blpop(ROLLOUTS)[1]

            rollout_data, uuid, name, result = _unserialize(rollout_bytes)
            latest_version = int(self.redis.get(VERSION_LATEST))

            # TODO log uuid?

            gamestates, actions, versions = rollout_data
            if not any(version < 0 and abs(version - latest_version) <= 1 for version in versions):
                continue

            gamestates = [GameState(gs) for gs in gamestates]
            obs_builder = self.obs_build_factory()
            rew_func = self.rew_build_factory()
            obs_builder.reset(gamestates[0])
            rew_func.reset(gamestates[0])
            buffers = [
                ExperienceBuffer()
                for _ in range(len(gamestates[0].players))
            ]
            last_actions = np.zeros((len(gamestates[0].players), 8))
            for s, gs in enumerate(gamestates):
                final = s == len(gamestates) - 1
                for i, player in enumerate(gs.players):
                    obs = obs_builder.build_obs(player, gs, last_actions[i])
                    if final:
                        rew = rew_func.get_final_reward(player, gs, last_actions[i])
                    else:
                        rew = rew_func.get_reward(player, gs, last_actions[i])
                    buffers[i].add_step(obs, actions[i][s], rew, final, None)
                    last_actions[i, :] = actions[i][s]

            ratings = []
            relevant_buffers = []
            for version, buffer in zip(versions, buffers):
                if version < 0 and abs(version - latest_version) <= 1:
                    rating = Rating(*_unserialize(self.redis.get(QUALITY_LATEST)))
                    relevant_buffers.append(buffer)
                else:
                    rating = Rating(*_unserialize(self.redis.lindex(QUALITIES, version)))
                ratings.append(rating)

            blue_players = sum(divmod(len(gamestates[0].players), 2))
            blue, orange = ratings[:blue_players], ratings[blue_players:]

            # In ranks lowest number is best, result=-1 is orange win, 0 tie, 1 blue
            r1, r2 = rate((blue, orange), ranks=(0, result))

            # Some trickery to handle same rating apearing multiple times, we just average their mus and sigmas
            versions = {}
            for rating, version in zip(r1 + r2, versions):
                versions.setdefault(version, []).append(rating)

            for version, ratings in versions.items():
                avg_rating = Rating((sum(r.mu for r in ratings) / len(ratings)),
                                    (sum(r.sigma for r in ratings) / len(ratings)))
                if version > 0:  # Old
                    self.redis.lset(QUALITIES, version, _serialize(tuple(avg_rating)))
                elif version == latest_version:
                    self.redis.set(QUALITY_LATEST, _serialize(tuple(avg_rating)))

            yield from relevant_buffers

    def _add_opponent(self, agent):
        # Add to list
        self.redis.rpush(OPPONENT_MODELS, agent)
        # Set quality
        ratings = [Rating(*_unserialize(v)) for v in self.redis.lrange(QUALITIES, 0, -1)]
        if ratings:
            mus = np.array([r.mu for r in ratings])
            conf = np.array([r.sigma for r in ratings])

            x = np.arange(len(mus))
            y = mus
            y_upper = mus + 2 * conf  # 95% confidence
            y_lower = mus - 2 * conf
            fig = go.Figure([
                go.Scatter(
                    x=x,
                    y=y,
                    line=dict(color='rgb(0,100,80)'),
                    mode='lines',
                    name="mu",
                    showlegend=False
                ),
                go.Scatter(
                    x=np.concatenate((x, x[::-1])),  # x, then x reversed
                    y=np.concatenate((y_upper, y_lower[::-1])),  # upper, then lower reversed
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',  # TODO same color as wandb run?
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    name="sigma",
                    showlegend=False
                ),
            ])

            fig.update_layout(title="Rating", xaxis_title="Iteration", yaxis_title="TrueSkill")

            self.logger.log({
                "qualities": fig,
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
        print("Top contributors:\n" + "\n".join(f"{c}: {n}" for c, n in self.contributors.most_common(5)))
        self.logger.log({
            "contributors": wandb.Table(columns=["name", "steps"], data=self.contributors.most_common())},
            commit=False
        )
        self.contributors.clear()

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

    def __init__(self, redis: Redis, name: str, match: Match, current_version_prob=.9, display_only=False):
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
        self.display_only = display_only

    def _get_opponent_indices(self, n):
        # Get qualities
        # sum priorities to have higher chance of selecting high sigma
        qualities = np.asarray([sum(_unserialize(v)) for v in self.redis.lrange(QUALITIES, 0, -1)])
        # Pick opponent
        probs = softmax(qualities / np.log(10))
        indices = np.random.choice(len(probs), n, p=probs)
        return indices

    def run(self):  # Mimics Thread
        """
        begin processing in already launched match and push to redis
        """
        n = 0
        t = Thread()
        t.start()
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
            agents = [(self.current_agent, latest_version)]  # Use at least one current agent

            if self.n_agents > 1:
                # Ensure final proportion is same
                adjusted_prob = (self.current_version_prob * self.n_agents - 1) / (self.n_agents - 1)
                n_old = np.random.binomial(n=self.n_agents - 1, p=1 - adjusted_prob)
                n_new = self.n_agents - n_old - 1
                old_versions = self._get_opponent_indices(n_old)
                for _ in range(n_new):
                    version = latest_version
                    selected_agent = self.current_agent
                    agents.append((selected_agent, version))

                for index in old_versions:
                    version = int(index)
                    selected_agent = _unserialize_model(self.redis.lindex(OPPONENT_MODELS, version))
                    agents.append((selected_agent, version))

            np.random.shuffle(agents)
            print("Generating rollout with versions:", [v for a, v in agents])

            rollouts, result = util.generate_episode(self.env, [agent for agent, version in agents])

            if not self.display_only:
                rollout_data = [
                    [encode_gamestate(info["state"]) for info in rollouts[0].infos],
                    [rollout.actions for rollout in rollouts],
                    [version for agent, version in agents]
                ]
                rollout_data = _serialize((rollout_data, self.uuid, self.name, result))
                t.join()
                t = Thread(target=lambda: self.redis.rpush(ROLLOUTS, rollout_data))
                t.start()
