from typing import List, Tuple

import numpy as np
import torch.distributions
from torch.distributions import Distribution, Categorical

from rlgym.gym import Gym
from rocket_learn.experience_buffer import ExperienceBuffer

from rocket_learn.agent import BaseAgent


def generate_episode(env: Gym, agents: List[BaseAgent]) -> (List[ExperienceBuffer], int):
    """
    create experience buffer data by interacting with the environment(s)
    """
    observations = env.reset()
    done = False

    rollouts = [
        ExperienceBuffer()
        for _ in range(len(agents))
    ]
    ep_rews = [0 for _ in range(len(agents))]
    with torch.no_grad():
        while not done:
            all_indices = []
            all_actions = []
            all_log_probs = []

            # if observation isn't a list, make it one so we don't iterate over the observation directly
            if not isinstance(observations, list):
                observations = [observations]

            for agent, obs in zip(agents, observations):
                dist = agent.get_action_distribution(obs)
                action_indices, log_prob = agent.get_action_indices(dist, deterministic=False, include_log_prob=True)
                actions = agent.get_action(action_indices)

                all_indices.append(action_indices)
                all_actions.append(actions)
                all_log_probs.append(log_prob)

            old_obs = observations
            observations, rewards, done, info = env.step(all_actions)
            if len(agents) <= 1:
                observations, rewards = [observations], [rewards]
            # Might be different if only one agent?
            for exp_buf, obs, act, rew, log_prob in zip(rollouts, old_obs, all_indices, rewards, all_log_probs):
                exp_buf.add_step(obs, act, rew, done, log_prob)

            for i in range(len(agents)):
                ep_rews[i] += rewards[i]

    return rollouts, info["result"]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class RunningMeanStd(object):  # From sb3
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
