from typing import List

from rlgym.gym import Gym
from experience_buffer import ExperienceBuffer


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
        # SOREN COMMENT:
        # Aren't we leaving that to the agents?


        actions = [agent.get_actions(obs) for obs, agent in zip(observations, agents)]
        log_probs = [agent.get_log_prob(action) for agent, actions in zip(agents, actions)]

        old_obs = observations
        observations, rewards, done, info = env.step(actions)
        # Might be different if only one agent?
        for exp_buf, obs, act, rew, log_prob in zip(rollouts, old_obs, actions, rewards, log_probs):
            exp_buf.add_step(obs, act, rew, done, log_prob)

    return rollouts
