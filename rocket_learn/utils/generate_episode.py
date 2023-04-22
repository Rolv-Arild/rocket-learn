from typing import List

import numpy as np
import torch
from rlgym.gym import Gym
from rlgym.utils.reward_functions.common_rewards import ConstantReward
from rlgym.utils.state_setters import DefaultState
from tqdm import tqdm

from rocket_learn.agent.policy import Policy
from rocket_learn.agent.pretrained_policy import HardcodedAgent
from rocket_learn.utils.experience_buffer import ExperienceBuffer
from rocket_learn.utils.dynamic_gamemode_setter import DynamicGMSetter


def generate_episode(env: Gym, policies, evaluate=False, scoreboard=None, progress=False) -> (
        List[ExperienceBuffer], int):
    """
    create experience buffer data by interacting with the environment(s)
    """
    if progress:
        progress = tqdm(unit=" steps")
    else:
        progress = None

    if evaluate:  # Change setup temporarily to play a normal game (approximately)
        from rlgym_tools.extra_terminals.game_condition import GameCondition  # tools is an optional dependency
        terminals = env._match._terminal_conditions  # noqa
        reward = env._match._reward_fn  # noqa
        game_condition = GameCondition(tick_skip=env._match._tick_skip,  # noqa
                                       seconds_per_goal_forfeit=10 * env._match._team_size,
                                       max_overtime_seconds=300,
                                       max_no_touch_seconds=60)
        env._match._terminal_conditions = [game_condition]  # noqa
        if isinstance(env._match._state_setter, DynamicGMSetter):  # noqa
            state_setter = env._match._state_setter.setter  # noqa
            env._match._state_setter.setter = DefaultState()  # noqa
        else:
            state_setter = env._match._state_setter  # noqa
            env._match._state_setter = DefaultState()  # noqa

        env._match._reward_fn = ConstantReward()  # noqa Save some cpu cycles

    if scoreboard is not None:
        random_resets = scoreboard.random_resets
        scoreboard.random_resets = not evaluate
    observations, info = env.reset(return_info=True)
    result = 0

    last_state = info['state']  # game_state for obs_building of other agents

    latest_policy_indices = [0 if isinstance(p, HardcodedAgent) else 1 for p in policies]
    # rollouts for all latest_policies
    rollouts = [
        ExperienceBuffer(infos=[info])
        for _ in range(sum(latest_policy_indices))
    ]

    b = o = 0
    with torch.no_grad():
        while True:
            all_indices = []
            all_actions = []
            all_log_probs = []

            # if observation isn't a list, make it one so we don't iterate over the observation directly
            if not isinstance(observations, list):
                observations = [observations]

            if not isinstance(policies[0], HardcodedAgent) and all(policy == policies[0] for policy in policies):
                policy = policies[0]
                if isinstance(observations[0], tuple):
                    obs = tuple(np.concatenate([obs[i] for obs in observations], axis=0)
                                for i in range(len(observations[0])))
                else:
                    obs = np.concatenate(observations, axis=0)
                dist = policy.get_action_distribution(obs)
                action_indices = policy.sample_action(dist)
                log_probs = policy.log_prob(dist, action_indices)
                actions = policy.env_compatible(action_indices)

                all_indices.extend(list(action_indices.numpy()))
                all_actions.extend(list(actions))
                all_log_probs.extend(list(log_probs.numpy()))
            else:
                index = 0
                for policy, obs in zip(policies, observations):
                    if isinstance(policy, HardcodedAgent):
                        actions = policy.act(last_state, index)

                        # make sure output is in correct format
                        if not isinstance(observations, np.ndarray):
                            actions = np.array(actions)

                        # TODO: add converter that takes normal 8 actions into action space
                        # actions = env._match._action_parser.convert_to_action_space(actions)

                        all_indices.append(None)
                        all_actions.append(actions)
                        all_log_probs.append(None)

                    elif isinstance(policy, Policy):
                        dist = policy.get_action_distribution(obs)
                        action_indices = policy.sample_action(dist)[0]
                        log_probs = policy.log_prob(dist, action_indices).item()
                        actions = policy.env_compatible(action_indices)

                        all_indices.append(action_indices.numpy())
                        all_actions.append(actions)
                        all_log_probs.append(log_probs)

                    else:
                        print(str(type(policy)) + " type use not defined")
                        assert False

                    index += 1

            # to allow different action spaces, pad out short ones to longest length (assume later unpadding in parser)
            # length = max([a.shape[0] for a in all_actions])
            # padded_actions = []
            # for a in all_actions:
            #     action = np.pad(a.astype('float64'), (0, length - a.size), 'constant', constant_values=np.NAN)
            #     padded_actions.append(action)
            #
            # all_actions = padded_actions
            # TEST OUT ABOVE TO DEAL WITH VARIABLE LENGTH

            all_actions = np.vstack(all_actions)
            old_obs = observations
            observations, rewards, done, info = env.step(all_actions)
            if len(policies) <= 1:
                observations, rewards = [observations], [rewards]

            # prune data that belongs to old agents
            old_obs = [a for i, a in enumerate(old_obs) if latest_policy_indices[i] == 1]
            all_indices = [d for i, d in enumerate(all_indices) if latest_policy_indices[i] == 1]
            rewards = [r for i, r in enumerate(rewards) if latest_policy_indices[i] == 1]
            all_log_probs = [r for i, r in enumerate(all_log_probs) if latest_policy_indices[i] == 1]

            assert len(old_obs) == len(all_indices), str(len(old_obs)) + " obs, " + str(len(all_indices)) + " ind"
            assert len(old_obs) == len(rewards), str(len(old_obs)) + " obs, " + str(len(rewards)) + " ind"
            assert len(old_obs) == len(all_log_probs), str(len(old_obs)) + " obs, " + str(len(all_log_probs)) + " ind"
            assert len(old_obs) == len(rollouts), str(len(old_obs)) + " obs, " + str(len(rollouts)) + " ind"

            # Might be different if only one agent?
            if not evaluate:  # Evaluation matches can be long, no reason to keep them in memory
                for exp_buf, obs, act, rew, log_prob in zip(rollouts, old_obs, all_indices, rewards, all_log_probs):
                    exp_buf.add_step(obs, act, rew, done, log_prob, info)

            if progress is not None:
                progress.update()
                igt = progress.n * env._match._tick_skip / 120  # noqa

                prog_str = f"{igt // 60:02.0f}:{igt % 60:02.0f} IGT"
                if evaluate:
                    prog_str += f", BLUE {b} - {o} ORANGE"
                progress.set_postfix_str(prog_str)

            if done:
                result += info["result"]
                if info["result"] > 0:
                    b += 1
                elif info["result"] < 0:
                    o += 1

                if not evaluate:
                    break
                elif game_condition.done:  # noqa
                    break
                else:
                    observations, info = env.reset(return_info=True)

            last_state = info['state']

    if scoreboard is not None:
        scoreboard.random_resets = random_resets  # noqa Checked above

    if progress is not None:
        progress.close()

    if evaluate:
        if isinstance(env._match._state_setter, DynamicGMSetter):  # noqa
            env._match._state_setter.setter = state_setter  # noqa
        else:
            env._match._state_setter = state_setter  # noqa
        env._match._terminal_conditions = terminals  # noqa
        env._match._reward_fn = reward  # noqa
        return result

    return rollouts, result
