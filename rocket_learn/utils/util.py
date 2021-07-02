from typing import List

from rlgym.gym import Gym
from rocket_learn.experience_buffer import ExperienceBuffer


def generate_episode(env: Gym, agents: list) -> List[ExperienceBuffer]:
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
        actions = [agent(obs) for obs, agent in zip(observations, agents)]
        old_obs = observations
        observations, rewards, done, info = env.step(actions)
        for exp_buf, obs, act, rew in zip(rollouts, old_obs, actions, rewards):  # Might be different if only one agent?
            exp_buf.add_step(obs, act, rew, done)

    return rollouts
