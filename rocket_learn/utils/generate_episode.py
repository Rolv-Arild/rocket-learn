import functools
from typing import List, Tuple

import numpy as np

from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.utils.gamestate_encoding import encode_gamestate
from rocket_learn.utils.dynamic_gamemode_setter import DynamicGMSetter
from rocket_learn.utils.truncated_condition import TruncatedCondition
from rocket_learn.utils.multi_env import MultiEnvManager
from rlgym_sim.gym import Gym
from rlgym_sim.utils.reward_functions.common_rewards import ConstantReward
from rlgym_sim.utils.state_setters import DefaultState
from tqdm import tqdm


def start_evaluation(idx, env):
    from rlgym_tools.extra_terminals.game_condition import GameCondition  # tools is an optional dependency

    n_agents = env._match._state_setter.blue + env._match._state_setter.orange  # noqa
    terminal = env._match._terminal_conditions  # noqa
    reward_fn = env._match._reward_fn  # noqa
    game_condition = GameCondition(tick_skip=env._game.tick_skip,  # noqa
                                   seconds_per_goal_forfeit=10 * n_agents // 2,
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

    env.temp_evaluation_info = (terminal, reward_fn, state_setter)  # Store for later


def end_evaluation(idx, env):
    terminal, reward_fn, state_setter = env.temp_evaluation_info
    del env.temp_evaluation_info
    if isinstance(env._match._state_setter, DynamicGMSetter):  # noqa
        env._match._state_setter.setter = state_setter  # noqa
    else:
        env._match._state_setter = state_setter  # noqa
    env._match._terminal_conditions = terminal  # noqa
    env._match._reward_fn = reward_fn  # noqa


def set_random_resets(idx, env, value):
    if env.scoreboard is not None:
        random_resets = env.scoreboard.random_resets
        env.scoreboard.random_resets = value
        return random_resets


def check_if_finished(idx, env, dones, is_finished, evaluate):
    # for env, done, info in zip(envs, dones, infos):
    game_condition = env._match._terminal_conditions[0]  # noqa
    if not is_finished[idx] and dones[idx] is not None and dones[idx] > 0:
        # result = info["result"]
        # results[e] += info["result"]
        # if result > 0:
        #     blue += 1
        # elif result < 0:
        #     orange += 1

        if not evaluate:
            return True
            # is_finished[idx] = True
        elif game_condition.done:
            return True
            # is_finished[idx] = True
        else:
            o, i = env.reset(return_info=True)
            env.awaiting_reset = False
            return o, i


def generate_episode(envs: Gym, policy_indices: List[Tuple["Policy", List[int]]], env_indices, evaluate=False,
                     progress=False, base_model=None, gpu_threshold=10) -> (List["ExperienceBuffer"], int):
    """
    create experience buffer data by interacting with the environment(s)
    """
    import torch

    from torch.distributions import Categorical
    from rocket_learn.agent.policy import Policy
    from rocket_learn.agent.pretrained_policy import HardcodedAgent

    if isinstance(envs, Gym) and not isinstance(envs, MultiEnvManager):
        envs = [envs]
    if progress:
        progress = tqdm(unit=" steps")
    else:
        progress = None

    n_agents = sum(len(i) for p, i in policy_indices)
    env_agents = [[a for a in range(len(env_indices)) if env_indices[a] == e] for e in range(len(envs))]

    if evaluate:  # Change setup temporarily to play a normal game (approximately)
        if isinstance(envs, list):
            for idx, env in enumerate(envs):
                start_evaluation(idx, env)
        elif isinstance(envs, MultiEnvManager):
            envs.send_function(start_evaluation)

    # else:
    #     for env in envs:
    #         if isinstance(env._match._reward_fn, ConstantReward):
    #             breakpoint()

    srr = functools.partial(set_random_resets, value=not evaluate)

    if isinstance(envs, list):
        all_random_resets = []
        for idx, env in enumerate(envs):
            rr = srr(idx, env)
            all_random_resets.append(rr)
    elif isinstance(envs, MultiEnvManager):
        all_random_resets = envs.send_function(srr)

    if isinstance(envs, list):
        last_states = []
        observations = []
        infos = []
        for env in envs:
            o, i = env.reset(return_info=True)

            last_state = i['state']  # game_state for obs_building of other agents
            i["numpy_state"] = encode_gamestate(last_state)

            last_states.append(last_state)
            observations.extend(o)
            infos.append(i)
    else:
        observations, infos = envs.reset()
        last_states = []
        for i in infos:
            last_state = i['state']  # game_state for obs_building of other agents
            i["numpy_state"] = encode_gamestate(last_state)
            last_states.append(last_state)
    results = [0] * len(envs)

    is_hardcoded = [False] * n_agents

    for policy, indices in policy_indices:
        if isinstance(policy, HardcodedAgent):
            for i in indices:
                is_hardcoded[i] = True

    # rollouts for all latest_policies
    rollouts = [
        ExperienceBuffer(infos=[infos[env_indices[i]]])
        for i in range(len(env_indices))
    ]

    is_finished = [False] * len(envs)

    blue = orange = 0
    with torch.no_grad():
        while True:
            all_action_indices = [None] * n_agents
            all_actions = [None] * n_agents
            all_log_probs = [None] * n_agents

            # if observation isn't a list, make it one so we don't iterate over the observation directly
            if not isinstance(observations, list):
                observations = [observations]

            for policy, indices in policy_indices:
                indices = [i for i in indices if not is_finished[env_indices[i]]]
                if len(indices) == 0:
                    continue
                if isinstance(policy, HardcodedAgent):
                    for index in indices:
                        last_state = last_states[env_indices[index]]
                        actions = policy.act(last_state, index)

                        # make sure output is in correct format
                        if not isinstance(observations, np.ndarray):
                            actions = np.array(actions)

                        # TODO: add converter that takes normal 8 actions into action space
                        # actions = env._match._action_parser.convert_to_action_space(actions)

                        # all_action_indices[index] = None
                        all_actions[index] = actions
                        # all_log_probs[index] = None
                elif isinstance(policy, Policy):
                    device = "cuda" if len(indices) >= gpu_threshold and torch.cuda.is_available() else "cpu"
                    policy.to(device)
                    if isinstance(observations[indices[0]], tuple):
                        obs = tuple(np.concatenate([observations[p][i] for p in indices], axis=0)
                                    for i in range(len(observations[indices[0]])))
                    else:
                        breakpoint()
                        obs = np.concatenate([observations[i] for i in indices], axis=0)

                    dist = policy.get_action_distribution(obs)
                    if policy.deterministic and base_model is not None and isinstance(dist, Categorical):
                        dist_base = base_model.get_action_distribution(obs)  # TODO mass run base model
                        a0 = dist_base.probs
                        a1 = dist.probs.cpu()
                        r = a1 * (torch.log(a1) - torch.log(a0))
                        dist = Categorical(logits=r)

                    action_indices = policy.sample_action(dist)

                    log_probs = policy.log_prob(dist, action_indices)
                    actions = policy.env_compatible(action_indices)

                    action_indices = action_indices.cpu().numpy()
                    log_probs = log_probs.cpu().numpy()
                    for i, idx in enumerate(indices):
                        all_action_indices[idx] = action_indices[i]
                        all_actions[idx] = actions[i]
                        all_log_probs[idx] = log_probs[i]
                else:
                    print(str(type(policy)) + " type use not defined")
                    assert False

            # to allow different action spaces, pad out short ones to longest length (assume later unpadding in parser)
            # length = max([a.shape[0] for a in all_actions])
            # padded_actions = []
            # for a in all_actions:
            #     action = np.pad(a.astype('float64'), (0, length - a.size), 'constant', constant_values=np.NAN)
            #     padded_actions.append(action)
            #
            # all_actions = padded_actions
            # TEST OUT ABOVE TO DEAL WITH VARIABLE LENGTH

            # all_actions = all_actions[sorted_idx]
            # all_action_indices = all_action_indices[sorted_idx]
            # all_log_probs = all_log_probs[sorted_idx]
            old_obs = observations

            if isinstance(envs, list):
                observations = [None] * n_agents
                rewards = [None] * n_agents
                dones = [None] * len(envs)
                infos = [None] * len(envs)
                k = 0
                for e, env in enumerate(envs):
                    new_k = k + len(env_agents[e])  # noqa
                    if is_finished[e]:
                        k = new_k
                        continue

                    env_actions = np.vstack(all_actions[k: new_k])
                    o, r, d, i = env.step(env_actions)

                    truncated = False
                    for terminal in env._match._terminal_conditions:  # noqa
                        if isinstance(terminal, TruncatedCondition):
                            truncated |= terminal.is_truncated(i["state"])
                    d = d + 2 * truncated

                    if isinstance(r, float):
                        o, r = [o], [r]

                    i["numpy_state"] = encode_gamestate(i["state"])  # So RocketSim GameState copying can be disabled

                    observations[k:new_k] = o
                    rewards[k:new_k] = r
                    dones[e] = d
                    infos[e] = i
                    # is_finished[e] |= d > 0
                    k = new_k
                assert k == len(env_indices)
            elif isinstance(envs, MultiEnvManager):
                observations, rewards, dones, infos = envs.step(all_actions)

            # # prune data that belongs to old agents
            # old_obs = [a for i, a in zip(all_indices, old_obs) if not is_hardcoded[i]]
            # all_action_indices = [d for i, d in zip(all_indices, all_action_indices) if not is_hardcoded[i]]
            # rewards = [r for i, r in zip(all_indices, rewards) if not is_hardcoded[i]]
            # all_log_probs = [r for i, r in zip(all_indices, all_log_probs) if not is_hardcoded[i]]
            #
            # assert len(old_obs) == len(all_action_indices), str(len(old_obs)) + " obs, " + str(
            #     len(all_action_indices)) + " ind"
            # assert len(old_obs) == len(rewards), str(len(old_obs)) + " obs, " + str(len(rewards)) + " ind"
            # assert len(old_obs) == len(all_log_probs), str(len(old_obs)) + " obs, " + str(len(all_log_probs)) + " ind"
            # assert len(old_obs) == len(rollouts), str(len(old_obs)) + " obs, " + str(len(rollouts)) + " ind"

            # Might be different if only one agent?
            if not evaluate:  # Evaluation matches can be long, no reason to keep them in memory
                for agent_idx, env_idx in enumerate(env_indices):
                    rew = rewards[agent_idx]
                    if rew is None:
                        continue
                    # if abs(rew) > 1:
                    #     print(rew)
                    obs = old_obs[agent_idx]
                    exp_buf = rollouts[agent_idx]

                    act = all_action_indices[agent_idx]
                    log_prob = all_log_probs[agent_idx]

                    done = dones[env_idx]
                    info = infos[env_idx]

                    exp_buf.add_step(obs, act, rew, done, log_prob, info)
                    if len(exp_buf.infos[0]["numpy_state"]) != len(exp_buf.infos[-1]["numpy_state"]):
                        breakpoint()
                # for env_idx, exp_buf, obs, act, rew, log_prob in zip(env_indices, rollouts, old_obs, all_action_indices,
                #                                                      rewards, all_log_probs):
                #     if obs is None:
                #         continue
                #     done = dones[env_idx]
                #     info = infos[env_idx]
                #     exp_buf.add_step(obs, act, rew, done, log_prob, info)

            if progress is not None and len(envs) == 1:
                env = envs[0]
                progress.update()
                igt = progress.n * env._game.tick_skip / 120  # noqa
                prog_str = f"{igt // 60:02.0f}:{igt % 60:02.0f} IGT"
                if evaluate:
                    prog_str += f", BLUE {blue} - {orange} ORANGE"
                progress.set_postfix_str(prog_str)

            if isinstance(envs, list):
                e = 0
                for env, done, info in zip(envs, dones, infos):
                    if not is_finished[e] and done is not None and done > 0:
                        results[e] += info["result"]
                        if info["result"] > 0:
                            blue += 1
                        elif info["result"] < 0:
                            orange += 1

                        if not evaluate:
                            if (info["result"] == 0) != (done >= 2):
                                breakpoint()
                            is_finished[e] = True
                        else:
                            game_condition = env._match._terminal_conditions[0]  # noqa
                            if game_condition.done:
                                is_finished[e] = True
                            else:
                                o, i = env.reset(return_info=True)
                                infos[e] = i
                                a = 0
                                for agent_idx, env_idx in enumerate(env_indices):
                                    if env_idx == e:
                                        observations[agent_idx] = o[a]
                                        a += 1
                    e += 1
            else:
                responses = envs.send_function(functools.partial(check_if_finished,
                                                                 dones=dones,
                                                                 is_finished=is_finished,
                                                                 evaluate=evaluate))
                for idx, resp in enumerate(responses):
                    if resp is not None:
                        if isinstance(resp, tuple):
                            o, i = resp
                            a = 0
                            for agent_idx, env_idx in enumerate(env_indices):
                                if env_idx == idx:
                                    observations[agent_idx] = o[a]
                                    a += 1
                        elif isinstance(resp, bool):
                            is_finished[idx] = resp

            if all(is_finished):
                break

            last_states = [info['state'] if info is not None else None for info in infos]

    # Correct the ordering to match indices
    rollouts = [[rollouts[i] for i in indices if not is_hardcoded[i]] for indices in env_agents]

    # breakpoint()

    # srr = functools.partial(set_random_resets, value=not evaluate)
    # if isinstance(envs, list):
    #     for idx, env in enumerate(envs):
    #         reset_random_resets(idx, env)
    # elif isinstance(envs, MultiEnvManager):
    #     envs.send_function(reset_random_resets)

    if evaluate:
        if isinstance(envs, list):
            for idx, env in enumerate(envs):
                end_evaluation(idx, env)
        elif isinstance(envs, MultiEnvManager):
            envs.send_function(end_evaluation)
        return results

    if progress is not None:
        progress.close()

    return rollouts, results
