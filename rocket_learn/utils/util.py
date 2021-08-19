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
    ep_rews = [0 for _ in range(len(agents))]
    with torch.no_grad():
        while not done:
            # TODO we need either:
            # - torch.distributions.Distribution
            # - (selected_action, <log_>prob) tuple
            # - logits for actions, 3*5+2*3=21 outputs
            # to calculate log_prob
            # SOREN COMMENT:
            # Aren't we leaving that to the agents?

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
    print(ep_rews)

    return rollouts


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
