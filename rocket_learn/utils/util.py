from typing import List

import numpy as np
import torch
import torch.distributions
from torch import nn

from rlgym.gym import Gym
from rocket_learn.agent.policy import Policy
from rocket_learn.experience_buffer import ExperienceBuffer


def generate_episode(env: Gym, policies: List[Policy]) -> (List[ExperienceBuffer], int):
    """
    create experience buffer data by interacting with the environment(s)
    """
    observations = env.reset()
    done = False

    rollouts = [
        ExperienceBuffer()
        for _ in range(len(policies))
    ]
    ep_rews = [0 for _ in range(len(policies))]
    with torch.no_grad():
        while not done:
            all_indices = []
            all_actions = []
            all_log_probs = []

            # if observation isn't a list, make it one so we don't iterate over the observation directly
            if not isinstance(observations, list):
                observations = [observations]

            for policy, obs in zip(policies, observations):
                dist = policy.get_action_distribution(obs)
                action_indices = policy.sample_action(dist, deterministic=False)
                log_prob = policy.log_prob(dist, action_indices).item()
                actions = policy.env_compatible(action_indices)

                all_indices.append(action_indices.numpy())
                all_actions.append(actions)
                all_log_probs.append(log_prob)

            all_actions = np.array(all_actions)
            old_obs = observations
            observations, rewards, done, info = env.step(all_actions)
            if len(policies) <= 1:
                observations, rewards = [observations], [rewards]
            # Might be different if only one agent?
            for exp_buf, obs, act, rew, log_prob in zip(rollouts, old_obs, all_indices, rewards, all_log_probs):
                exp_buf.add_step(obs, act, rew, done, log_prob)

            for i in range(len(policies)):
                ep_rews[i] += rewards[i]

    result = info["result"]
    # result = 0 if abs(info["state"].ball.position[1]) < BALL_RADIUS else (2 * (info["state"].ball.position[1] > 0) - 1)

    return rollouts, result


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class SplitLayer(nn.Module):
    def __init__(self, splits=None):
        super().__init__()
        if splits is not None:
            self.splits = splits
        else:
            self.splits = (3, 3, 3, 3, 3, 2, 2, 2)

    def forward(self, x):
        return torch.split(x, self.splits, dim=-1)
