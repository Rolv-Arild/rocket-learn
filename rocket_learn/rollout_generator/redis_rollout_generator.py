import functools
import itertools
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
import psutil
import wandb
# from matplotlib.axes import Axes
# from matplotlib.figure import Figure
from gym.vector.utils import CloudpickleWrapper
from redis import Redis
from redis.exceptions import ResponseError
from rlgym.utils import ObsBuilder, RewardFunction
from rlgym.utils.action_parsers import ActionParser
from trueskill import Rating, rate, SIGMA
import plotly.graph_objs as go

from rlgym.envs import Match
from rlgym.gamelaunch import LaunchPreference
from rlgym.gym import Gym
from rlgym.utils.gamestates import GameState
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.rollout_generator.base_rollout_generator import BaseRolloutGenerator
from rocket_learn.utils import util
from rocket_learn.utils.batched_obs_builder import BatchedObsBuilder
from rocket_learn.utils.util import encode_gamestate, probability_NvsM, softmax

# Constants for consistent key lookup
QUALITIES = "qualities"
N_UPDATES = "num-updates"
SAVE_FREQ = "save-freq"

MODEL_LATEST = "model-latest"
VERSION_LATEST = "model-version"

ROLLOUTS = "rollout"
OPPONENT_MODELS = "opponent-models"
WORKER_IDS = "worker-ids"
CONTRIBUTORS = "contributors"
_ALL = (
    QUALITIES, N_UPDATES, SAVE_FREQ, MODEL_LATEST, VERSION_LATEST, ROLLOUTS, OPPONENT_MODELS,
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


def encode_buffers(buffers: List[ExperienceBuffer], strict=False, send_rewards=True):
    if strict:
        states = np.asarray([encode_gamestate(info["state"]) for info in buffers[0].infos])
        actions = np.asarray([buffer.actions for buffer in buffers])
        log_probs = np.asarray([buffer.log_probs for buffer in buffers])
        if send_rewards:
            rewards = np.asarray([buffer.rewards for buffer in buffers])
            return states, actions, log_probs, rewards
        return states, actions, log_probs
    else:
        return [
            (buffer.meta, buffer.observations, buffer.actions, buffer.rewards, buffer.dones, buffer.log_probs)
            for buffer in buffers
        ]


def decode_buffers(enc_buffers, encoded, obs_build_factory=None, rew_func_factory=None, act_parse_factory=None):
    if encoded:
        if len(enc_buffers) == 3:
            game_states, actions, log_probs = enc_buffers
            rewards = None
        elif len(enc_buffers) == 4:
            game_states, actions, log_probs, rewards = enc_buffers
        else:
            raise ValueError

        obs_builder = obs_build_factory()
        act_parser = act_parse_factory()

        if isinstance(obs_builder, BatchedObsBuilder):
            assert rewards is not None
            obs = obs_builder.batched_build_obs(game_states[:-1])
            prev_actions = act_parser.parse_actions(actions.reshape((-1,) + actions.shape[2:]).copy(), None).reshape(
                actions.shape[:2] + (8,))
            prev_actions = np.concatenate((np.zeros((actions.shape[0], 1, 8)), prev_actions[:, :-1]), axis=1)
            obs_builder.add_actions(obs, prev_actions)
            dones = np.zeros_like(rewards, dtype=bool)
            dones[-1, :] = True
            buffers = [
                ExperienceBuffer(observations=[obs[i]], actions=actions[i], rewards=rewards[i], dones=dones[i],
                                 log_probs=log_probs[i])
                for i in range(len(obs))
            ]
            return buffers

        game_states = [GameState(gs.tolist()) for gs in game_states]
        rew_func = rew_func_factory()
        obs_builder.reset(game_states[0])
        rew_func.reset(game_states[0])
        buffers = [
            ExperienceBuffer(infos=[{"state": game_states[0]}])
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
                if rewards is None:
                    if final:
                        rew = rew_func.get_final_reward(player, gs, env_actions[s][i])
                    else:
                        rew = rew_func.get_reward(player, gs, env_actions[s][i])
                else:
                    rew = rewards[i][s]
                buffers[i].add_step(old_obs[i], actions[i][s], rew, final, log_probs[i][s], {"state": gs})
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
            clear=True,
            mmr_min_episode_length=150
    ):
        self.tot_bytes = 0
        self.redis = redis
        self.logger = logger

        # TODO saving/loading
        if clear:
            self.redis.delete(*_ALL)
            self.redis.set(N_UPDATES, 0)
        else:
            if self.redis.exists(ROLLOUTS) > 0:
                self.redis.delete(ROLLOUTS)
            self.redis.decr(VERSION_LATEST, 2)  # In case of reload from old version, don't let current seep in

        self.redis.set(SAVE_FREQ, save_every)
        self.contributors = Counter()  # No need to save, clears every iteration
        self.obs_build_func = obs_build_factory
        self.rew_func_factory = rew_func_factory
        self.act_parse_factory = act_parse_factory
        self.mmr_min_episode_length = mmr_min_episode_length

    @staticmethod
    def _process_rollout(rollout_bytes, latest_version, obs_build_func, rew_build_func, act_build_func):
        rollout_data, versions, uuid, name, result, encoded = _unserialize(rollout_bytes)

        if any(version < 0 and abs(version - latest_version) > 1 for version in versions):
            return

        buffers = decode_buffers(rollout_data, encoded, obs_build_func, rew_build_func, act_build_func)
        return buffers, versions, uuid, name, result

    def _update_ratings(self, name, versions, buffers, latest_version, result):
        ratings = []
        relevant_buffers = []
        for version, buffer in itertools.zip_longest(versions, buffers):
            if version < 0:
                if abs(version - latest_version) <= 1:
                    relevant_buffers.append(buffer)
                    self.contributors[name] += buffer.size()
                else:
                    return []
            else:
                rating = Rating(*_unserialize(self.redis.lindex(QUALITIES, version)))
                ratings.append(rating)

        # Only old versions, calculate MMR
        if len(ratings) == len(versions) and len(buffers) == 0:
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
                self.redis.lset(QUALITIES, version, _serialize(tuple(avg_rating)))

        return relevant_buffers

    def generate_rollouts(self) -> Iterator[ExperienceBuffer]:
        while True:
            latest_version = int(self.redis.get(VERSION_LATEST))
            data = self.redis.blpop(ROLLOUTS)[1]
            self.tot_bytes += len(data)
            res = self._process_rollout(
                data, latest_version,
                self.obs_build_func, self.rew_func_factory, self.act_parse_factory
            )
            if res is not None:
                buffers, versions, uuid, name, result = res
                relevant_buffers = self._update_ratings(name, versions, buffers, latest_version, result)
                yield from relevant_buffers

        # futures = []
        # cpus = psutil.cpu_count(logical=False)
        # with ProcessPoolExecutor(cpus) as ex:
        #     while True:
        #         # Kinda scuffed ngl
        #         if len(futures) > 0 and futures[0].done():
        #             res = futures.pop(0).result()
        #             if res is not None:
        #                 latest_version = int(self.redis.get(VERSION_LATEST))
        #                 buffers, versions, uuid, name, result = res
        #                 relevant_buffers = self._update_ratings(name, versions, buffers, latest_version, result)
        #                 yield from relevant_buffers
        #         elif len(futures) < 2 * cpus:
        #             latest_version = int(self.redis.get(VERSION_LATEST))
        #             data = self.redis.blpop(ROLLOUTS)[1]
        #             self.tot_bytes += len(data)
        #             futures.append(ex.submit(
        #                 RedisRolloutGenerator._process_rollout,
        #                 data,
        #                 latest_version,
        #                 CloudpickleWrapper(self.obs_build_func),
        #                 CloudpickleWrapper(self.rew_func_factory),
        #                 CloudpickleWrapper(self.act_parse_factory)
        #             ))

    def _plot_ratings(self, ratings):
        if len(ratings) <= 0:
            return
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

    def _add_opponent(self, agent):
        # Add to list
        self.redis.rpush(OPPONENT_MODELS, agent)
        # Set quality
        ratings = [Rating(*_unserialize(v)) for v in self.redis.lrange(QUALITIES, 0, -1)]
        if ratings:
            quality = Rating(ratings[-1].mu, SIGMA)
        else:
            quality = Rating(0, 1)  # First (basically random) agent is initialized at 0
        self.redis.rpush(QUALITIES, _serialize(tuple(quality)))

    def update_parameters(self, new_params):
        """
        update redis (and thus workers) with new model data and save data as future opponent
        :param new_params: new model parameters
        """
        model_bytes = _serialize_model(new_params)
        self.redis.set(MODEL_LATEST, model_bytes)
        self.redis.decr(VERSION_LATEST)

        print("Top contributors:\n" + "\n".join(f"{c}: {n}" for c, n in self.contributors.most_common(5)))
        self.logger.log({
            "contributors": wandb.Table(columns=["name", "steps"], data=self.contributors.most_common())},
            commit=False
        )
        self._plot_ratings([Rating(*_unserialize(v)) for v in self.redis.lrange(QUALITIES, 0, -1)])
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

    def __init__(self, redis: Redis, name: str, match: Match,
                 current_version_prob=.8, evaluation_prob=0.01, sigma_target=1,
                 display_only=False, send_gamestates=True, jit_compile=False,
                 deterministic_old_prob=0.5):
        # TODO model or config+params so workers can recreate just from redis connection?
        self.redis = redis
        self.name = name

        self.display_only = display_only

        self.current_agent = _unserialize_model(self.redis.get(MODEL_LATEST))
        self.current_version_prob = current_version_prob
        self.evaluation_prob = evaluation_prob
        self.sigma_target = sigma_target
        self.send_gamestates = send_gamestates
        self.jit_compile = jit_compile
        self.deterministic_old_prob = deterministic_old_prob

        # **DEFAULT NEEDS TO INCORPORATE BASIC SECURITY, THIS IS NOT SUFFICIENT**
        self.uuid = str(uuid4())
        self.redis.rpush(WORKER_IDS, self.uuid)

        if not self.display_only:
            print("Started worker", self.uuid, "on host", self.redis.connection_pool.connection_kwargs.get("host"),
                  "under name", name)  # TODO log instead
        else:
            print("Streaming mode set. Running silent.")

        self.match = match
        self.env = Gym(match=self.match, pipe_id=os.getpid(), launch_preference=LaunchPreference.EPIC_LOGIN_TRICK,
                       use_injector=True)
        self.n_agents = self.match.agents

    def _get_opponent_indices(self, n_new, n_old):
        if n_old == 0:
            return [-1] * n_new
        # Get qualities
        ratings = [Rating(*_unserialize(v)) for v in self.redis.lrange(QUALITIES, 0, -1)]

        if n_new == 0:  # Evaluation game, try to find agents with high sigma
            sigmas = np.array([r.sigma for r in ratings])
            probs = np.clip(sigmas - self.sigma_target, a_min=0, a_max=None)
            s = probs.sum()
            if s == 0:
                n_new = np.random.randint(1, n_old)
                return self._get_opponent_indices(n_new, n_old - n_new)
            probs /= s
            versions = [np.random.choice(len(ratings), p=probs)]
            target_rating = ratings[versions[0]]
            n_old -= 1
        else:
            versions = [-1] * n_new
            target_rating = ratings[-1]

        # Calculate 1v1 win prob against target
        # All the agents included should hold their own (at least approximately)
        # This is to prevent unrealistic scenarios,
        # like for instance ratings of [100, 0] vs [100, 0], which is technically fair but not useful
        probs = np.zeros(len(ratings))
        for i, rating in enumerate(ratings):
            p = probability_NvsM([rating], [target_rating])
            probs[i] = (p * (1 - p)) ** (2 / (n_old + n_new))  # Be a little bit less strict the more players there are
        probs /= probs.sum()

        old_versions = np.random.choice(len(probs), size=n_old, p=probs).tolist()
        versions += old_versions

        # Then calculate the full matchup, with just permutations of the selected versions (weighted by fairness)
        matchups = []
        qualities = []
        for perm in itertools.permutations(versions):
            it_ratings = [ratings[v] for v in perm]
            mid = len(it_ratings) // 2
            p = probability_NvsM(it_ratings[:mid], it_ratings[mid:])
            matchups.append(perm)
            qualities.append(p * (1 - p))  # From AlphaStar
        qualities = np.array(qualities)
        k = np.random.choice(len(matchups), p=qualities / qualities.sum())
        return matchups[k]

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
                if r < self.evaluation_prob:
                    n_old = self.n_agents
                elif (r - self.evaluation_prob) / (1 - self.evaluation_prob) > self.current_version_prob:
                    n_old = np.random.randint(low=1, high=self.n_agents)

            n_new = self.n_agents - n_old
            versions = self._get_opponent_indices(n_new, n_old)
            agents = []
            for version in versions:
                if version == -1:
                    agents.append(self.current_agent)
                else:
                    selected_agent = self._get_past_model(version)
                    if np.random.random() < self.deterministic_old_prob:
                        selected_agent.deterministic = True
                    agents.append(selected_agent)
            versions = [v if v != -1 else latest_version for v in versions]

            encode = self.send_gamestates
            if all(v >= 0 for v in versions) and not self.display_only:
                for agent in agents:
                    agent.deterministic = False  # maybe True instead?
                print("Running evaluation game with versions:", versions)
                result = util.generate_episode(self.env, agents, evaluate=True)
                rollouts = []
                print("Evaluation finished, goal differential:", result)
                encode = False
            else:
                if not self.display_only:
                    print("Generating rollout with versions:", versions)

                rollouts, result = util.generate_episode(self.env, agents, evaluate=False)
                if len(rollouts[0].observations) <= 1:
                    rollouts, result = util.generate_episode(self.env, agents, evaluate=False)

                state = rollouts[0].infos[-2]["state"]
                goal_speed = np.linalg.norm(state.ball.linear_velocity) * 0.036  # kph
                str_result = ('+' if result > 0 else "") + str(result)
                post_stats = f"Rollout finished after {len(rollouts[0].observations)} steps, result was {str_result}"
                if result != 0:
                    post_stats += f", goal speed: {goal_speed:.2f} kph"

                if not self.display_only:
                    print(post_stats)

            if not self.display_only:
                rollout_data = encode_buffers(rollouts, strict=encode)  # TODO change
                # sanity_check = decode_buffers(rollout_data, encode,
                #                               lambda: self.match._obs_builder,
                #                               lambda: self.match._reward_fn,
                #                               lambda: self.match._action_parser)
                rollout_bytes = _serialize((rollout_data, versions, self.uuid, self.name, result,
                                            encode))
                # while True:
                t.join()

                def send():
                    n_items = self.redis.rpush(ROLLOUTS, rollout_bytes)
                    if n_items >= 1000:
                        print("Had to limit rollouts. Learner may have have crashed, or is overloaded")
                        self.redis.ltrim(ROLLOUTS, -100, -1)

                t = Thread(target=send)
                t.start()
                time.sleep(0.01)
