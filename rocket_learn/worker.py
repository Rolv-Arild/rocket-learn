import os
import pickle
import time

import numpy as np
from rlgym.envs import Match
from rlgym.gym import Gym

from redis import Redis
import msgpack
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM

QUALITIES = "qualities"
MODEL_LATEST = "model-latest"
MODEL_N = "model-{}"
ROLLOUTS = "rollout"


def update_model(redis, state_dict_dump, version):
    redis.set(Keys.LATEST_MODEL, state_dict_dump)
    redis.set(Keys.LATEST_VERSION, version)


def add_opponent(redis, state_dict_dump):
    # Add to list
    redis.rpush(Keys.OP_MODELS, state_dict_dump)
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


def worker(epic_rl_path, current_version_prob=0.8, **match_args):
    redis = Redis()
    match = Match(**match_args)
    env = Gym(match=match, pipe_id=os.getpid(), path_to_rl=epic_rl_path, use_injector=True)
    n_agents = match.agents
    while True:
        current_agent = msgpack.loads(redis.get(MODEL_LATEST))

        # TODO customizable past agent selection, should team only be same agent?
        agents = [(current_agent, MODEL_LATEST)]  # Use at least one current agent

        if n_agents > 1:
            adjusted_prob = (current_version_prob * n_agents - 1) / (n_agents - 1)  # Ensure final proportion is same
            for i in range(n_agents - 1):
                is_current = np.random.random() < adjusted_prob
                if not is_current:
                    index, prob = get_opponent_index(redis)
                    version = MODEL_N.format(index)
                    selected_agent = msgpack.loads(redis.get(version))
                else:
                    prob = current_version_prob
                    version = MODEL_LATEST
                    selected_agent = current_agent

                agents.append((selected_agent, version, prob))

        np.random.shuffle(agents)

        observations = env.reset()
        done = False
        rollouts = [[] for agent, version in agents if version == MODEL_LATEST]
        while not done:
            actions = [agent.get_action(agent.get_action_distribution(obs))
                       for (agent, version), obs in zip(agents, observations)]
            observations, rewards, done, info = env.step(actions)

            state = info["state"]

            n = 0
            for i, ((agent, version, prob), player) in enumerate(zip(agents, state.players)):
                result = info["result"]
                if player.TEAM_NUM == ORANGE_TEAM:
                    result = -result

                update_opponent_quality(redis, version, prob, result * 0.01)

                if version == MODEL_LATEST:
                    rollouts[n].append((observations[i], actions[i], rewards[i], result))
                    n += 1
        redis.rpush(ROLLOUTS, *(msgpack.dumps(rollout) for rollout in rollouts))


def redis_rollout_generator():
    redis = Redis()
    while True:
        item = redis.lpop(ROLLOUTS)
        if item is not None:
            rollout = msgpack.loads(item)
            yield rollout
        else:
            time.sleep(10)


def learner(rollout_generator, n_rollouts=36, learning_rate=3e-4, gamma=0.9):
    while True:
        rollouts = []
        while len(rollouts) < n_rollouts:
            try:
                rollout = next(rollout_generator)
                rollouts.append(rollout)
            except StopIteration:
                return

        # TODO batch, horizon, PPO
