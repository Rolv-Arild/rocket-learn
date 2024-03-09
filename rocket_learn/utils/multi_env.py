import multiprocessing as mp
import os
from typing import Callable

import numpy as np
from gym import Env

from rocket_learn.utils.gamestate_encoding import encode_gamestate
from rocket_learn.utils.truncated_condition import TruncatedCondition


def _env_process(idx, env_creator, remote, parent_remote):
    env = env_creator()
    env.awaiting_reset = True
    parent_remote.close()

    remote.send(("init", None))  # Signal initialization complete
    while True:
        action = remote.recv()
        if action[0] == "reset":
            obs_info = env.reset(return_info=True)
            env.awaiting_reset = False
            remote.send(("reset_result", obs_info))
        elif action[0] == "step":
            if not env.awaiting_reset:
                obs, reward, done, info = env.step(np.vstack(action[1]))
                truncated = False
                for terminal in env._match._terminal_conditions:  # noqa
                    if isinstance(terminal, TruncatedCondition):
                        truncated |= terminal.is_truncated(info["state"])
                if done or truncated:
                    env.awaiting_reset = True
                    # env.is_finished = True
                info["numpy_state"] = encode_gamestate(info["state"])

                remote.send(("step_result", (obs, reward, done + 2 * truncated, info)))
            else:
                # raise ValueError("Tried to step in env that is awaiting reset.")
                remote.send(("step_result", None))
        elif action[0] == "close":
            env.close()
            remote.send(("close", None))
            break
        elif action[0] == "func":
            func = action[1]
            result = func(idx, env)
            remote.send(("func_result", result))


class MultiEnvManager(Env):
    def __init__(self, env_creator: Callable[[], Env], num_envs):
        self.env_creator = env_creator
        self.num_envs = num_envs

        self.agent_counts = np.zeros(num_envs, dtype=int)
        self.processes = []

        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # Fork is not a thread safe method (see issue #217)
        # but is more user friendly (does not require to wrap the code in
        # a `if __name__ == "__main__":`)
        forkserver_available = "forkserver" in mp.get_all_start_methods()
        start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.parent_conn, self.child_conn = zip(*[ctx.Pipe() for _ in range(num_envs)])

        for idx in range(num_envs):
            process = ctx.Process(target=_env_process,
                                  args=(idx, env_creator, self.child_conn[idx], self.parent_conn[idx]),
                                  daemon=True)
            process.start()
            self.processes.append(process)
            self.child_conn[idx].close()
            assert self.parent_conn[idx].recv()[0] == "init"

        if cuda_devices is None:
            del os.environ['CUDA_VISIBLE_DEVICES']
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices

        # self._start_processes()

    # def _start_processes(self):
    #     for process in self.processes:
    #         process.start()
    #
    #     for conn in self.child_conn:
    #         assert conn.recv()[0] == "init"

    def reset(self):
        for conn in self.parent_conn:
            conn.send(("reset", None))  # Signal reset to environments

        obs_list = []
        info_list = []
        for conn in self.parent_conn:
            obs, info = conn.recv()[1]
            obs_list.append(obs)
            info_list.append(info)
        self.agent_counts[:] = [len(obs) for obs in obs_list]
        return sum(obs_list, start=[]), info_list

    def step(self, actions):
        i = 0
        for idx in range(self.num_envs):
            agent_count = self.agent_counts[idx]
            env_actions = actions[i:i + agent_count]
            # print(type(env_actions), env_actions, env_actions.dtype)
            self.parent_conn[idx].send(("step", env_actions))
            i += agent_count

        all_obs = []
        all_rewards = []
        all_dones = []
        all_infos = []
        for i, conn in enumerate(self.parent_conn):
            step_result = conn.recv()[1]
            if step_result is None:
                all_obs += [None] * self.agent_counts[i]
                all_rewards += [None] * self.agent_counts[i]
                all_dones += [None]
                all_infos += [None]
            else:
                observations, rewards, done, info = step_result
                all_obs += observations
                all_rewards += rewards
                all_dones += [done]
                all_infos += [info]
        return all_obs, all_rewards, all_dones, all_infos

    def send_function(self, func):
        for conn in self.parent_conn:
            conn.send(("func", func))

        results = [conn.recv()[1] for conn in self.parent_conn]
        return results

    def close(self):
        for conn in self.parent_conn:
            conn.send(("close", None))

        for process in self.processes:
            process.join()

    def render(self, mode="human"):
        raise NotImplementedError

    def __len__(self):
        return self.num_envs
