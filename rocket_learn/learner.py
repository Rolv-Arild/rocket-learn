import os
import pickle
import time

import numpy as np

from redis import Redis
import msgpack

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import multiprocessing as mp

from worker import worker
from experience_buffer import ExperienceBuffer

from typing import Any
import cloudpickle


class CloudpickleWrapper:
    """
    Copied from SB3

    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

    :param var: the variable you wish to wrap for pickling with cloudpickle
    """

    def __init__(self, var: Any):
        self.var = var

    def __getstate__(self) -> Any:
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var: Any) -> None:
        self.var = cloudpickle.loads(var)

class Learner:
    def __init__(self, rl_exe_path, algorithm, log_dir, match_arg_list, n_envs=1):
        self.logger = SummaryWriter(log_dir)

        self.algorithm = algorithm
        self.algorithm.set_logger(self.logger)

        self.buffer = ExperienceBuffer()
        self.workers = []

        #**DEFAULT NEEDS TO INCORPORATE BASIC SECURITY, THIS IS NOT SUFFICIENT**
        self.redis = Redis(host='127.0.0.1', port=6379, db=0)

        #assert n_envs == len(match_arg_list()) or len(match_arg_list()) == 1, \
        #    "number of environments must match number of match arguments or be length 1"

        # SOREN COMMENT: need to support multiple match args so different envs can do
        # different things
        self.buildWorkers(rl_exe_path, match_arg_list, n_envs=n_envs)


    def learn(self, n_rollouts = 36):
        # SOREN COMMENT: what's a good model number scheme and do you account for continuing
        # training from a saved model?

        # SOREN COMMENT: this is an ugly way of doing it
        traj_count = 0
        while True:
            val = self.recieve_worker_data()
            self.buffer.add_step(**val)
            traj_count += 1

            if traj_count >= n_rollouts:
                self.__calculate__(self.buffer)

                self.buffer.clear()
                traj_count = 0

                # SOREN COMMENT:
                # Add base algorithm that requires implementation of this method
                # also, figure out best transmission object
                state_dict = algorithm.policy_dict_list()

                worker.update_model(self.redit, state_dict, model_version)
                model_version += 1

                # SOREN COMMENT: add in launching extra workers after X amount of time


    # pulled from rolv's wrapper and SB3
    def buildWorkers(self,  match_args_func, path_to_epic_rl, n_envs=1, wait_time=60):
        env_fns = [match_args_func for _ in range(n_envs)]

        forkserver_available = "forkserver" in mp.get_all_start_methods()
        start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        for work_remote, remote, env_fn in zip(work_remotes, remotes, env_fns):
            #args = (path_to_epic_rl, 1, match_args_func) # ** test this, probably wrong
            args = ()#(work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=worker, args=args, daemon=True)
            process.start()

            self.workers.append(process)
            work_remote.close()
            time.sleep(wait_time)



    def recieve_worker_data(self):
        while True:
            item = self.redis.lpop(ROLLOUTS)
            if item is not None:
                rollout = msgpack.loads(item)
                yield rollout
            else:
                time.sleep(10)



    def __calculate__(self):
        #apply PPO now but separate so we can refactor to allow different algorithm types
        self.algorithm.calculate(self.buffer)



#this should probably be in its own file
class PPO:
    def __init__(self, actor, critic, lr_actor = 3e-4, lr_critic = 3e-4, gamma = 0.9, epochs = 1):
        self.actor = actor
        self.critic = critic

        #hyperparameters
        self.epochs = epochs
        self.gamma = gamma
        self.gae_lambda = 0
        self.batch_size = 512
        self.clip_range = .2
        self.ent_coef = 1
        self.vf_coef = 1
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr_actor},
            {'params': self.critic.parameters(), 'lr': lr_critic}
        ])

    def set_logger(self, logger):
        self.logger = logger

    def policy_pickle(self):
        networks = [self.actor.get_state_dict(), self.critic.get_state_dict()]
        return networks

    def evaluate_actions(self):
        ## **TODO**
        return -1, -1

    def calculate(self, buffer: ExperienceBuffer):
        values = self.critic(buffer)
        buffer_size = buffer.size()

        #totally stole this section from
        #https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8
        #I am not attached to it, make it better if you'd like
        returns = []
        gae = 0
        for i in reversed(range(buffer_size)):
            delta = buffer.rewards[i] + self.gamma * values[i + 1] * buffer.dones[i] - values[i]
            gae = delta + self.gamma * lmbda * buffer.dones[i] * gae
            returns.insert(0, gae + values[i])

        advantages = np.array(returns) - values[:-1]
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
        #returns is also called references?

        for e in range(self.epochs):
            # SOREN COMMENT:
            # I'm going with SB3 PPO implementation cause we'll have a reference

            # this is mostly pulled from sb3
            for i, rollout in enumerate(self.buffer.generate_rollouts(self.batch_size)):
                #this should probably be consolidated into buffer
                adv = self.advantages[i:i+batch_size]

                log_prob, entropy = self.evaluate_actions()
                ratio = torch.exp(log_prob - rollout.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = adv * ratio
                policy_loss_2 = adv * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # **If we want value clipping, add it here**
                value_loss = F.mse_loss(returns, values)

                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with torch.no_grad():
                    log_ratio = log_prob - old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()

                self.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()


                #self.logger write here to log results