from typing import List

import numpy as np
import torch.distributions
from torch.distributions import Distribution, Categorical

from rlgym.gym import Gym
from rocket_learn.experience_buffer import ExperienceBuffer

from rocket_learn.agent import BaseAgent


def generate_episode(env: Gym, agents: List[BaseAgent]) -> List[ExperienceBuffer]:
    observations = env.reset()
    done = False

    rollouts = [
        ExperienceBuffer()
        for _ in range(len(agents))
    ]

    while not done:
        all_indices = []
        all_actions = []
        all_log_probs = []

        #if observation isn't a list, make it one so we don't iterate over the observation directly
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
        ex = env.action_space.sample()
        cleanedAction = int(all_actions[0].item())
        observations, rewards, done, info = env.step(cleanedAction)


        #if done:
            #print(np.sum(exp_buf.rewards))

        if len(agents) <= 1:
            observations, rewards = [observations], [rewards]

        # Might be different if only one agent?
        for exp_buf, obs, act, rew, log_prob in zip(rollouts, old_obs, all_indices, rewards, all_log_probs):
            exp_buf.add_step(obs, act, rew, done, log_prob)

    return rollouts


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
