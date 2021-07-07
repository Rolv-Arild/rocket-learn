from typing import List

import numpy as np

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
        # TODO we need either:
        # - torch.distributions.Distribution
        # - (selected_action, <log_>prob) tuple
        # - logits for actions, 3*5+2*3=21 outputs
        # to calculate log_prob
        # SOREN COMMENT:
        # Aren't we leaving that to the agents?

        actions_probs = [agent.get_action_with_log_prob(obs) for obs, agent in zip(observations, agents)]
        actions, log_probs = zip(*actions_probs)

        old_obs = observations
        observations, rewards, done, info = env.step(actions)
        if len(agents) <= 1:
            observations, rewards = [observations], [rewards]
        # Might be different if only one agent?
        for exp_buf, obs, act, rew, log_prob in zip(rollouts, old_obs, actions, rewards, log_probs):
            exp_buf.add_step(obs, act, rew, done, log_prob)

    return rollouts


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
