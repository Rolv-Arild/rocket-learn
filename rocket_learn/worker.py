import os
import pickle
import time
import io

import numpy as np
from rlgym.envs import Match
from rlgym.gym import Gym

import torch
import torch.nn as nn

from redis import Redis
import msgpack
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM

from experience_buffer import ExperienceBuffer
from utils import util
from simple_agents import PPOAgent

# SOREN COMMENT:
# need to move all keys into dedicated file?
QUALITIES = "qualities"
MODEL_ACTOR_LATEST = "model-actor-latest"
MODEL_CRITIC_LATEST = "model-critic-latest"
MODEL_N = "model-{}"
ROLLOUTS = "rollout"
VERSION_LATEST = "model-version"


#DELETE THESE AFTER TESTING
state_dim = 67
action_dim = 8

#example pytorch stuff, delete later
actor = nn.Sequential(
    nn.Linear(state_dim, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, action_dim),
    nn.Softmax(dim=-1)
)

# critic
critic = nn.Sequential(
    nn.Linear(state_dim, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
)


def update_model(redis, agent, version):
    if redis.exists(MODEL_ACTOR_LATEST) > 0:
        redis.delete(MODEL_ACTOR_LATEST)
    if redis.exists(MODEL_CRITIC_LATEST) > 0:
        redis.delete(MODEL_CRITIC_LATEST)
    if redis.exists(VERSION_LATEST) > 0:
        redis.delete(VERSION_LATEST)

    actor_bytes = pickle.dumps(agent.actor.state_dict())
    critic_bytes = pickle.dumps(agent.critic.state_dict())

    redis.set(MODEL_ACTOR_LATEST, actor_bytes)
    redis.set(MODEL_CRITIC_LATEST, critic_bytes)
    redis.set(VERSION_LATEST, version)
    print("done setting")


def add_opponent(redis, state_dict_dump):
    # Add to list
    redis.rpush(OP_MODELS, state_dict_dump)
    # Set quality
    qualities = [float(v) for v in redis.lrange(QUALITIES, 0, -1)]
    if qualities:
        quality = max(qualities)
    else:
        quality = 0
    redis.rpush(QUALITIES, quality)


def _softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_opponent_index(redis):
    # Get qualities
    qualities = np.asarray([float(v) for v in redis.lrange(QUALITIES, 0, -1)])
    # Pick opponent
    probs = _softmax(qualities)
    index = np.random.choice(len(probs), p=probs)
    return index, probs[index]


def update_opponent_quality(redis, index, prob, rate):
    # Calculate delta
    n = redis.llen(QUALITIES)
    delta = rate / (n * prob)
    # lua script to read and update atomically
    redis.eval('''
        local q = tonumber(redis.call('LINDEX', KEYS[1], KEYS[2]))
        local delta = tonumber(ARGV[1])
        local new_q = q + delta
        return redis.call('LSET', KEYS[1], KEYS[2], new_q)
        ''', 2, QUALITIES, index, delta)


def Worker(): #epic_rl_path, current_version_prob=0.8, **match_args):
    epic_rl_path="E:\\EpicGames\\rocketleague\\Binaries\\Win64\\RocketLeague.exe"
    current_version_prob=.8

    current_agent = PPOAgent(actor, critic)

    redis = Redis()
    match = Match()#**match_args)
    env = Gym(match=match, pipe_id=os.getpid(), path_to_rl=epic_rl_path, use_injector=True)
    n_agents = match.agents

    # ROLV COMMENT:
    # MODEL_LATEST is the current parameters from the latest policy update.
    # Past agents (saved every couple iterations) are selected randomly based on their quality.
    # We could cache so we save some communication overhead in case it reuses agents.
    # I just copied OpenAI which uses past agents 20% of the time, and latest parameters otherwise.

    while True:
        actor_dict = pickle.loads(redis.get(MODEL_ACTOR_LATEST))
        current_agent.actor.load_state_dict(actor_dict)

        critic_dict = pickle.loads(redis.get(MODEL_CRITIC_LATEST))
        current_agent.critic.load_state_dict(critic_dict)

        # TODO customizable past agent selection, should team only be same agent?
        agents = [(current_agent, MODEL_ACTOR_LATEST)]  # Use at least one current agent

        if n_agents > 1:
            # Ensure final proportion is same
            adjusted_prob = (current_version_prob * n_agents - 1) / (n_agents - 1)
            for i in range(n_agents - 1):
                is_current = np.random.random() < adjusted_prob
                if not is_current:
                    index, prob = get_opponent_index(redis)
                    version = MODEL_N.format(index)
                    selected_agent = msgpack.loads(redis.get(version))
                else:
                    prob = current_version_prob
                    version = VERSION_LATEST
                    selected_agent = current_agent

                agents.append((selected_agent, version, prob))

        np.random.shuffle(agents)

        rollouts = util.generate_episode(env, agents)

        redis.rpush(ROLLOUTS, *(msgpack.dumps(rollout) for rollout in rollouts))
