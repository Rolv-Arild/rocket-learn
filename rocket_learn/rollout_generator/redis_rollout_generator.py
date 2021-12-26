import functools
import os
import random
import zlib
from concurrent.futures import ProcessPoolExecutor
from threading import Thread

import cloudpickle as pickle
import time
from collections import Counter
from typing import Iterator, Callable, List
from uuid import uuid4

import msgpack
import msgpack_numpy as m
import numpy as np
# import matplotlib.pyplot  # noqa
import wandb
# from matplotlib.axes import Axes
# from matplotlib.figure import Figure
from gym.vector.utils import CloudpickleWrapper
from redis import Redis
from redis.exceptions import ResponseError
from rlgym.utils import ObsBuilder, RewardFunction
from rlgym.utils.action_parsers import ActionParser
from trueskill import Rating, rate
import plotly.graph_objs as go

from rlgym.envs import Match
from rlgym.gamelaunch import LaunchPreference
from rlgym.gym import Gym
from rlgym.utils.gamestates import GameState
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.rollout_generator.base_rollout_generator import BaseRolloutGenerator
from rocket_learn.utils import util
from rocket_learn.utils.util import encode_gamestate, probability_NvsM

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
CONTRIBUTORS = "contributors"
_ALL = (
    QUALITIES, N_UPDATES, SAVE_FREQ, MODEL_LATEST, ROLLOUTS, VERSION_LATEST, QUALITY_LATEST, OPPONENT_MODELS,
    WORKER_IDS, CONTRIBUTORS)

m.patch()


# Helper methods for easier changing of byte conversion
def _serialize(obj):
    return zlib.compress(msgpack.packb(obj))


def _unserialize(obj):
    return msgpack.unpackb(zlib.decompress(obj))


def _serialize_model(mdl):
    device = next(mdl.parameters()).device  # Must be a better way right?
    mdl_bytes = pickle.dumps(mdl.cpu())
    mdl.to(device)
    return mdl_bytes


def _unserialize_model(buf):
    agent = pickle.loads(buf)
    return agent


def encode_buffers(buffers: List[ExperienceBuffer], strict=False):
    if strict:
        states = np.asarray([encode_gamestate(info["state"]) for info in buffers[0].infos])
        actions = np.asarray([buffer.actions for buffer in buffers])
        log_probs = np.asarray([buffer.log_probs for buffer in buffers])
        return states, actions, log_probs
    else:
        return [
            (buffer.meta, buffer.observations, buffer.actions, buffer.rewards, buffer.dones, buffer.log_probs)
            for buffer in buffers
        ]


def decode_buffers(enc_buffers, obs_build_factory=None, rew_func_factory=None, act_parse_factory=None):
    if len(enc_buffers) == 3:
        game_states, actions, log_probs = enc_buffers
        game_states = [GameState(gs.tolist()) for gs in game_states]
        obs_builder = obs_build_factory()
        rew_func = rew_func_factory()
        act_parser = act_parse_factory()
        obs_builder.reset(game_states[0])
        rew_func.reset(game_states[0])
        buffers = [
            ExperienceBuffer()
            for _ in range(len(game_states[0].players))
        ]

        env_actions = [
            act_parser.parse_actions(actions[:, s, :].copy(), game_states[s])
            for s in range(actions.shape[1])
        ]

        obss = [obs_builder.build_obs(p, game_states[0], np.zeros(8))
                for i, p in enumerate(game_states[0].players)]
        for s, gs in enumerate(game_states[1:]):
            final = s == len(game_states) - 2
            old_obs = obss
            obss = []
            for i, player in enumerate(gs.players):
                obs = obs_builder.build_obs(player, gs, env_actions[s][i])
                if final:
                    rew = rew_func.get_final_reward(player, gs, env_actions[s][i])
                else:
                    rew = rew_func.get_reward(player, gs, env_actions[s][i])
                buffers[i].add_step(old_obs[i], actions[i][s], rew, final, log_probs[i][s], None)
                obss.append(obs)

        return buffers
    else:
        buffers = []
        for enc_buffer in enc_buffers:
            meta, obs, actions, rews, dones, log_probs = enc_buffer
            buffers.append(
                ExperienceBuffer(meta=meta, observations=obs, actions=actions,
                                 rewards=rews, dones=dones, log_probs=log_probs)
            )
        return buffers


class RedisRolloutGenerator(BaseRolloutGenerator):
    """
    Rollout generator in charge of sending commands to workers via redis
    """

    def __init__(
            self,
            redis: Redis,
            obs_build_factory: Callable[[], ObsBuilder],
            rew_func_factory: Callable[[], RewardFunction],
            act_parse_factory: Callable[[], ActionParser],
            save_every=10,
            logger=None,
            clear=True
    ):
        self.tot_bytes = 0
        self.redis = redis
        self.logger = logger

        # TODO saving/loading
        if clear:
            self.redis.delete(*_ALL)
            self.redis.set(N_UPDATES, 0)
            self.redis.set(QUALITY_LATEST, _serialize((0, 1)))
        else:
            if self.redis.exists(ROLLOUTS) > 0:
                self.redis.delete(ROLLOUTS)
            self.redis.decr(VERSION_LATEST, 2)  # In case of reload from old version, don't let current seep in

        self.redis.set(SAVE_FREQ, save_every)
        self.contributors = Counter()  # No need to save, clears every iteration
        self.obs_build_func = obs_build_factory
        self.rew_func_factory = rew_func_factory
        self.act_parse_factory = act_parse_factory

    @staticmethod
    def _process_rollout(rollout_bytes, latest_version, obs_build_func, rew_build_func, act_build_func):
        rollout_data, versions, uuid, name, result = _unserialize(rollout_bytes)

        if not any(version < 0 and abs(version - latest_version) <= 1 for version in versions):
            return

        buffers = decode_buffers(rollout_data, obs_build_func, rew_build_func, act_build_func)
        return buffers, versions, uuid, name, result

    def _update_ratings(self, name, versions, buffers, latest_version, result):
        ratings = []
        relevant_buffers = []
        for version, buffer in zip(versions, buffers):
            if version < 0:
                if abs(version - latest_version) <= 1:
                    rating = Rating(*_unserialize(self.redis.get(QUALITY_LATEST)))
                    relevant_buffers.append(buffer)
                    self.contributors[name] += buffer.size()
                else:
                    return []
            else:
                rating = Rating(*_unserialize(self.redis.lindex(QUALITIES, version)))
            ratings.append(rating)

        blue_players = sum(divmod(len(ratings), 2))
        blue = tuple(ratings[:blue_players])  # Tuple is important
        orange = tuple(ratings[blue_players:])

        # In ranks lowest number is best, result=-1 is orange win, 0 tie, 1 blue
        r1, r2 = rate((blue, orange), ranks=(0, result))

        # Some trickery to handle same rating appearing multiple times, we just average their new mus and sigmas
        ratings_versions = {}
        for rating, version in zip(r1 + r2, versions):
            ratings_versions.setdefault(version, []).append(rating)

        for version, ratings in ratings_versions.items():
            avg_rating = Rating((sum(r.mu for r in ratings) / len(ratings)),
                                (sum(r.sigma ** 2 for r in ratings) ** 0.5 / len(ratings)))  # Average vars
            if version >= 0:  # Old
                self.redis.lset(QUALITIES, version, _serialize(tuple(avg_rating)))
            elif version == latest_version:
                self.redis.set(QUALITY_LATEST, _serialize(tuple(avg_rating)))

        return relevant_buffers

    def generate_rollouts(self) -> Iterator[ExperienceBuffer]:
        # while True:
        #     relevant_buffers = self._process_rollout()
        #     if relevant_buffers is not None:
        #         yield from relevant_buffers
        futures = []
        with ProcessPoolExecutor(8) as ex:
            while True:
                # Kinda scuffed ngl

                if len(futures) > 0 and futures[0].done():
                    res = futures.pop(0).result()
                    if res is not None:
                        latest_version = int(self.redis.get(VERSION_LATEST))
                        buffers, versions, uuid, name, result = res
                        relevant_buffers = self._update_ratings(name, versions, buffers, latest_version, result)
                        yield from relevant_buffers
                elif len(futures) < os.cpu_count():
                    latest_version = int(self.redis.get(VERSION_LATEST))
                    data = self.redis.blpop(ROLLOUTS)[1]
                    self.tot_bytes += len(data)
                    futures.append(ex.submit(
                        RedisRolloutGenerator._process_rollout,
                        data,
                        latest_version,
                        CloudpickleWrapper(self.obs_build_func),
                        CloudpickleWrapper(self.rew_func_factory),
                        CloudpickleWrapper(self.act_parse_factory)
                    ))

    def _add_opponent(self, agent):
        # Add to list
        self.redis.rpush(OPPONENT_MODELS, agent)
        # Set quality
        ratings = [Rating(*_unserialize(v)) for v in self.redis.lrange(QUALITIES, 0, -1)]
        if ratings:
            mus = np.array([r.mu for r in ratings])
            mus = mus - mus[0]
            sigmas = np.array([r.sigma for r in ratings])
            # sigmas[1:] = (sigmas[1:] ** 2 + sigmas[0] ** 2) ** 0.5

            x = np.arange(len(mus))
            y = mus
            y_upper = mus + 2 * sigmas  # 95% confidence
            y_lower = mus - 2 * sigmas

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
            # quality = Rating(ratings[-1].mu, 2 * ratings[-1].sigma)
            quality = Rating(_unserialize(self.redis.get(QUALITY_LATEST)))  # Same mu, reset sigma
            self.redis.set(QUALITY_LATEST, _serialize(tuple(Rating(_unserialize(self.redis.get(QUALITY_LATEST))[0]))))
        else:
            quality = Rating(0, 1)  # First (typically random) agent is initialized at 0
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

        # TODO Idea: workers send name to identify who contributed rollouts,
        # keep track of top rollout contributors (each param update and total)
        # Also UID to keep track of current number of contributing workers?
        print("Top contributors:\n" + "\n".join(f"{c}: {n}" for c, n in self.contributors.most_common(5)))
        self.logger.log({
            "contributors": wandb.Table(columns=["name", "steps"], data=self.contributors.most_common())},
            commit=False
        )
        tot_contributors = self.redis.hgetall(CONTRIBUTORS)
        tot_contributors = Counter({name: int(count) for name, count in tot_contributors.items()})
        tot_contributors += self.contributors
        if tot_contributors:
            self.redis.hset(CONTRIBUTORS, mapping=tot_contributors)
        self.contributors.clear()

        self.logger.log({"rollout_bytes": self.tot_bytes}, commit=False)
        self.tot_bytes = 0

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

    def __init__(self, redis: Redis, name: str, match: Match, current_version_prob=.9, display_only=False,
                 send_gamestates=True):
        # TODO model or config+params so workers can recreate just from redis connection?
        self.redis = redis
        self.name = name

        self.current_agent = _unserialize_model(self.redis.get(MODEL_LATEST))
        self.current_version_prob = current_version_prob
        self.send_gamestates = send_gamestates

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

    def _get_opponent_indices(self, n_new, n_old):
        setup = [False] * n_new + [True] * n_old
        random.shuffle(setup)
        # Get qualities
        # sum priorities to have higher chance of selecting high sigma
        ratings = [Rating(*_unserialize(v)) for v in self.redis.lrange(QUALITIES, 0, -1)]
        best_matchup = None
        best_quality = -1
        for it in range(round(len(ratings) ** 0.5)):
            matchup = np.full(len(setup), -1)
            versions = np.random.choice(len(ratings), size=n_old)
            matchup[setup] = versions
            it_ratings = [ratings[v] for v in matchup]
            mid = len(it_ratings) // 2
            quality = abs(0.5 - probability_NvsM(it_ratings[:mid], it_ratings[mid:]))
            if quality > best_quality:
                best_matchup = [int(v) for v in matchup]
                best_quality = quality
        return best_matchup

    @functools.lru_cache(maxsize=8)
    def _get_past_model(self, version):
        return _unserialize_model(self.redis.lindex(OPPONENT_MODELS, version))

    def run(self):  # Mimics Thread
        """
        begin processing in already launched match and push to redis
        """
        n = 0
        latest_version = None
        t = Thread()
        t.start()
        while True:
            # Get the most recent version available
            available_version = self.redis.get(VERSION_LATEST)
            if available_version is None:
                time.sleep(1)
                continue  # Wait for version to be published (not sure if this is necessary?)
            available_version = int(available_version)

            # Only try to download latest version when new
            if latest_version != available_version:
                model_bytes = self.redis.get(MODEL_LATEST)
                if model_bytes is None:
                    time.sleep(1)
                    continue  # This is maybe not necessary? Can't hurt to leave it in.
                latest_version = available_version
                updated_agent = _unserialize_model(model_bytes)
                self.current_agent = updated_agent

            n += 1

            # TODO customizable past agent selection, should team only be same agent?
            n_old = 0
            if self.n_agents > 1:
                r = np.random.random()
                if r > self.current_version_prob:
                    n_old = np.random.binomial(n=self.n_agents, p=0.5)

            n_new = self.n_agents - n_old
            versions = self._get_opponent_indices(n_new, n_old)
            agents = []
            for version in versions:
                if version == -1:
                    agents.append((self.current_agent, latest_version))
                else:
                    selected_agent = self._get_past_model(version)
                    agents.append((selected_agent, version))

            print("Generating rollout with versions:", [v for a, v in agents])

            rollouts, result = util.generate_episode(self.env, [agent for agent, version in agents])

            state = rollouts[0]["info"]["state"]
            goal_speed = np.linal.norm(state.ball.linear_velocity) * 0.036  # kph
            str_result = ('+' + result if result > 0 else str(result))
            post_stats = f"Rollout finished after {len(rollouts[0])} steps, result was {str_result}"
            if result != 0:
                post_stats += f", goal speed: {goal_speed} kph"
            print(post_stats)

            if not self.display_only:
                rollout_data = encode_buffers(rollouts, strict=self.send_gamestates)  # TODO change
                # sanity_check = decode_buffers(rollout_data,
                #                               lambda: self.match._obs_builder,
                #                               lambda: self.match._reward_fn,
                #                               lambda: self.match._action_parser)
                versions = [version for agent, version in agents]
                rollout_bytes = _serialize((rollout_data, versions, self.uuid, self.name, result))
                t.join()

                def send():
                    n_items = self.redis.rpush(ROLLOUTS, rollout_bytes)
                    if n_items >= 1000:
                        print("Had to limit rollouts. Learner may have have crashed, or is overloaded")
                        self.redis.ltrim(ROLLOUTS, -100, -1)

                t = Thread(target=send)
                t.start()
