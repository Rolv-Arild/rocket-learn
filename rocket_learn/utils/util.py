from typing import List, Any

import numpy as np
import torch
import torch.distributions
from rlgym.gym import Gym
from rlgym.utils import ObsBuilder
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions.common_rewards import ConstantReward
from rlgym.utils.state_setters import DefaultState
from torch import nn

from rocket_learn.agent.policy import Policy
from rocket_learn.agent.pretrained_policy import PretrainedDiscretePolicy, HardcodedAgent

from rocket_learn.experience_buffer import ExperienceBuffer
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition


def generate_episode(env: Gym, policies, evaluate=False) -> (List[ExperienceBuffer], int):
    """
    create experience buffer data by interacting with the environment(s)
    """
    if evaluate:  # Change setup temporarily to play a normal game (approximately)
        from rlgym_tools.extra_terminals.game_condition import GameCondition  # tools is an optional dependency
        state_setter = env._match._state_setter  # noqa
        terminals = env._match._terminal_conditions  # noqa
        reward = env._match._reward_fn  # noqa
        game_condition = GameCondition(tick_skip=env._match._tick_skip,
                                       forfeit_spg_limit=10 * env._match._team_size)  # noqa
        env._match._terminal_conditions = [game_condition, GoalScoredCondition()]  # noqa
        env._match._state_setter = DefaultState()  # noqa
        env._match._reward_fn = ConstantReward()  # noqa Save some cpu cycles

    observations, info = env.reset(return_info=True)
    result = 0

    last_state = info['state']  # game_state for obs_building of other agents

    latest_policy_indices = [0 if isinstance(p, HardcodedAgent) else 1 for p in policies]
    # rollouts for all latest_policies
    rollouts = [
        ExperienceBuffer(infos=[info])
        for _ in range(sum(latest_policy_indices))
    ]

    with torch.no_grad():
        while True:
            all_indices = []
            all_actions = []
            all_log_probs = []

            # if observation isn't a list, make it one so we don't iterate over the observation directly
            if not isinstance(observations, list):
                observations = [observations]

            for policy, obs in zip(policies, observations):
                if isinstance(policy, HardcodedAgent):
                    actions = policy.act(last_state)

                    #make sure output is in correct format
                    if not isinstance(observations, np.ndarray):
                        actions = np.array(actions)

                    # TODO: add converter that takes normal 8 actions into action space
                    #actions = env._match._action_parser.convert_to_action_space(actions)

                    all_indices.append(None)
                    all_actions.append(actions)
                    all_log_probs.append(None)

                elif isinstance(policy, Policy):
                    dist = policy.get_action_distribution(obs)
                    action_indices = policy.sample_action(dist, deterministic=False)
                    log_probs = policy.log_prob(dist, action_indices).item()
                    actions = policy.env_compatible(action_indices)

                    all_indices.append(action_indices.numpy())
                    all_actions.append(actions)
                    all_log_probs.append(log_probs)

                else:
                    print(str(type(policy)) + " type use not defined")
                    assert False

            all_actions = np.array(all_actions)
            old_obs = observations
            observations, rewards, done, info = env.step(all_actions)
            if len(policies) <= 1:
                observations, rewards = [observations], [rewards]

            #prune data that belongs to old agents
            old_obs = [a for i, a in enumerate(old_obs) if latest_policy_indices[i] == 1]
            all_indices = [d for i, d in enumerate(all_indices) if latest_policy_indices[i] == 1]
            rewards = [r for i, r in enumerate(rewards) if latest_policy_indices[i] == 1]
            all_log_probs = [r for i, r in enumerate(all_log_probs) if latest_policy_indices[i] == 1]

            # Might be different if only one agent?
            for exp_buf, obs, act, rew, log_prob in zip(rollouts, old_obs, all_indices, rewards, all_log_probs):
                exp_buf.add_step(np.squeeze(obs), np.squeeze(act), rew, done, log_prob, info)

            if done:
                result += info["result"]
                if not evaluate:
                    break
                elif game_condition.done:  # noqa
                    break
                else:
                    observations, info = env.reset(return_info=True)

            last_state = info['state']

    if evaluate:
        env._match._terminal_conditions = terminals  # noqa
        env._match._state_setter = state_setter  # noqa
        env._match._reward_fn = reward  # noqa
        return result

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
            self.splits = (3,) * 5 + (2,) * 3

    def forward(self, x):
        return torch.split(x, self.splits, dim=-1)


def encode_gamestate(state: GameState):
    state_vals = [0, state.blue_score, state.orange_score]
    state_vals += state.boost_pads.tolist()

    for bd in (state.ball, state.inverted_ball):
        state_vals += bd.position.tolist()
        state_vals += bd.linear_velocity.tolist()
        state_vals += bd.angular_velocity.tolist()

    for p in state.players:
        state_vals += [p.car_id, p.team_num]
        for cd in (p.car_data, p.inverted_car_data):
            state_vals += cd.position.tolist()
            state_vals += cd.quaternion.tolist()
            state_vals += cd.linear_velocity.tolist()
            state_vals += cd.angular_velocity.tolist()
        state_vals += [
            p.match_goals,
            p.match_saves,
            p.match_shots,
            p.match_demolishes,
            p.boost_pickups,
            p.is_demoed,
            p.on_ground,
            p.ball_touched,
            p.has_flip,
            p.boost_amount
        ]
    return state_vals


# TODO AdvancedObs should be supported by default, use stack instead of cat
class ExpandAdvancedObs(AdvancedObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        return np.reshape(
            super(ExpandAdvancedObs, self).build_obs(player, state, previous_action),
            (1, -1)
        )


def probability_NvsM(team1_ratings, team2_ratings, env=None):
    from trueskill import global_env
    # Trueskill extension, source: https://github.com/sublee/trueskill/pull/17
    """Calculates the win probability of the first team over the second team.
    :param team1_ratings: ratings of the first team participants.
    :param team2_ratings: ratings of another team participants.
    :param env: the :class:`TrueSkill` object.  Defaults to the global
                environment.
    """
    if env is None:
        env = global_env()

    team1_mu = sum(r.mu for r in team1_ratings)
    team1_sigma = sum((env.beta ** 2 + r.sigma ** 2) for r in team1_ratings)
    team2_mu = sum(r.mu for r in team2_ratings)
    team2_sigma = sum((env.beta ** 2 + r.sigma ** 2) for r in team2_ratings)

    x = (team1_mu - team2_mu) / np.sqrt(team1_sigma + team2_sigma)
    probability_win_team1 = env.cdf(x)
    return probability_win_team1
