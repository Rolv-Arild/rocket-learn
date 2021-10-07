from typing import List

import numpy as np
import torch
import torch.distributions
from torch import nn

from rlgym.gym import Gym
from rlgym.utils.gamestates import GameState, PhysicsObject, PlayerData
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


def _encode_player(pd: PlayerData):
    return np.concatenate((
        _encode_physics_object(pd.car_data),
        _encode_physics_object(pd.inverted_car_data),
        np.array([
            pd.match_goals, pd.match_saves, pd.match_shots, pd.match_demolishes,
            pd.boost_pickups, pd.is_demoed, pd.on_ground, pd.ball_touched,
            pd.has_flip, pd.boost_amount, pd.car_data, pd.team_num
        ])
    ))


def _encode_physics_object(po: PhysicsObject):
    return np.concatenate((
        po.position,
        po.quaternion,
        po.linear_velocity,
        po.angular_velocity
    ))


def encode_gamestate(state: GameState):
    return np.concatenate(
        (
            np.array([np.nan, state.blue_score, state.orange_score]),
            state.boost_pads,
            _encode_physics_object(state.ball),
            _encode_physics_object(state.inverted_ball)
        ) +
        tuple(
            _encode_player(p) for p in state.players
        )
    )
