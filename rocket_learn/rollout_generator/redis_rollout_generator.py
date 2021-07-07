import os
import pickle
import time
from typing import Generator, Iterator

import numpy as np
from redis import Redis
from torch import nn

from rlgym.envs import Match
from rlgym.gym import Gym
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.rollout_generator.base_rollout_generator import BaseRolloutGenerator

# Hopefully we only need this one file, so this is where it belongs
from rocket_learn.simple_agents import PPOAgent
from rocket_learn.utils import util
from rocket_learn.utils.util import softmax

QUALITIES = "qualities"
MODEL_ACTOR_LATEST = "model-actor-latest"
MODEL_CRITIC_LATEST = "model-critic-latest"
MODEL_N = "model-{}"
ROLLOUTS = "rollout"
VERSION_LATEST = "model-version"
OP_MODELS = "opponent_models"


class RedisRolloutGenerator(BaseRolloutGenerator):
    def __init__(self, save_every=10):
        # **DEFAULT NEEDS TO INCORPORATE BASIC SECURITY, THIS IS NOT SUFFICIENT**
        self.redis = Redis(host='127.0.0.1', port=6379, db=0)
        self.n_updates = 0
        self.save_every = save_every

    def generate_rollouts(self) -> Iterator[ExperienceBuffer]:
        while True:
            rollout = self.redis.lpop(ROLLOUTS)
            if rollout is not None:  # Assuming nil is converted to None by py-redis
                yield rollout
            time.sleep(1)  # Don't DOS Redis

    def _update_model(self, agent, version):  # TODO same as update_parameters?
        if self.redis.exists(MODEL_ACTOR_LATEST) > 0:
            self.redis.delete(MODEL_ACTOR_LATEST)
        if self.redis.exists(MODEL_CRITIC_LATEST) > 0:
            self.redis.delete(MODEL_CRITIC_LATEST)
        if self.redis.exists(VERSION_LATEST) > 0:
            self.redis.delete(VERSION_LATEST)

        actor_bytes = pickle.dumps(agent.actor.state_dict())
        critic_bytes = pickle.dumps(agent.critic.state_dict())

        self.redis.set(MODEL_ACTOR_LATEST, actor_bytes)
        self.redis.set(MODEL_CRITIC_LATEST, critic_bytes)
        self.redis.set(VERSION_LATEST, version)
        print("done setting")

    def _add_opponent(self, state_dict_dump):  # TODO use
        # Add to list
        self.redis.rpush(OP_MODELS, state_dict_dump)
        # Set quality
        qualities = [float(v) for v in self.redis.lrange(QUALITIES, 0, -1)]
        if qualities:
            quality = max(qualities)
        else:
            quality = 0
        self.redis.rpush(QUALITIES, quality)

    def update_parameters(self, new_params):
        self.redis.set(VERSION_LATEST, new_params)
        if self.n_updates % self.save_every == 0:
            self.redis.set(MODEL_N.format(self.n_updates // self.save_every), new_params)


class RedisRolloutWorker:  # Provides RedisRolloutGenerator with rollouts via a Redis server
    def __init__(self, epic_rl_path, match_args=None, current_version_prob=.8, actor=None, critic=None):
        # example pytorch stuff, delete later
        self.state_dim = 67
        self.action_dim = 8
        self.actor = actor
        if actor is None:
            self.actor = nn.Sequential(
                nn.Linear(self.state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, self.action_dim),
                nn.Softmax(dim=-1)
            )

        self.critic = critic
        if critic is None:
            critic = nn.Sequential(
                nn.Linear(self.state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

        self.current_agent = PPOAgent(actor, critic)
        self.current_version_prob = current_version_prob

        # **DEFAULT NEEDS TO INCORPORATE BASIC SECURITY, THIS IS NOT SUFFICIENT**
        self.redis = Redis(host='127.0.0.1', port=6379, db=0)
        self.match = Match(**(match_args if match_args is not None else {}))
        self.env = Gym(match=self.match, pipe_id=os.getpid(), path_to_rl=epic_rl_path, use_injector=True)
        self.n_agents = self.match.agents

    def _get_opponent_index(self):
        # Get qualities
        qualities = np.asarray([float(v) for v in self.redis.lrange(QUALITIES, 0, -1)])
        # Pick opponent
        probs = softmax(qualities)
        index = np.random.choice(len(probs), p=probs)
        return index, probs[index]

    def _update_opponent_quality(self, index, prob, rate):  # TODO use
        # Calculate delta
        n = self.redis.llen(QUALITIES)
        delta = rate / (n * prob)
        # lua script to read and update atomically
        self.redis.eval('''
            local q = tonumber(redis.call('LINDEX', KEYS[1], KEYS[2]))
            local delta = tonumber(ARGV[1])
            local new_q = q + delta
            return redis.call('LSET', KEYS[1], KEYS[2], new_q)
            ''', 2, QUALITIES, index, delta)

    def run(self):  # Mimics Thread
        while True:
            actor_dict = pickle.loads(self.redis.get(MODEL_ACTOR_LATEST))
            self.current_agent.actor.load_state_dict(actor_dict)

            critic_dict = pickle.loads(self.redis.get(MODEL_CRITIC_LATEST))
            self.current_agent.critic.load_state_dict(critic_dict)

            # TODO customizable past agent selection, should team only be same agent?
            agents = [(self.current_agent, MODEL_ACTOR_LATEST, 1.)]  # Use at least one current agent

            if self.n_agents > 1:
                # Ensure final proportion is same
                adjusted_prob = (self.current_version_prob * self.n_agents - 1) / (self.n_agents - 1)
                for i in range(self.n_agents - 1):
                    is_current = np.random.random() < adjusted_prob
                    if not is_current:
                        index, prob = self._get_opponent_index()
                        version = MODEL_N.format(index)
                        selected_agent = pickle.loads(self.redis.get(version))
                    else:
                        prob = self.current_version_prob
                        version = VERSION_LATEST
                        selected_agent = self.current_agent

                    agents.append((selected_agent, version, prob))

            np.random.shuffle(agents)

            rollouts = util.generate_episode(self.env, [agent for agent, version, prob in agents])

            self.redis.rpush(ROLLOUTS, *(pickle.dumps(rollout) for rollout in rollouts))


