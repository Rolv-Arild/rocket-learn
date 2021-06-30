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

from rocket_learn.worker import worker
from rocket_learn.experience_buffer import ExperienceBuffer



#example pytorch stuff, delete later
actor = nn.Sequential(
    nn.Linear(state_dim, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, action_dim),
    nn.Softmax(dim=-1)
)

# critic
critic = nn.Sequential(
    nn.Linear(state_dim, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
)


def get_match_args():
    return dict(
        game_speed=100,
        random_resets=True,
        self_play=SELF_PLAY,
        team_size=1,
        obs_builder=AdvancedObs(),
        terminal_conditions=[TimeoutCondition(600), GoalScoredCondition()],  # 500 = 25 seconds in game
        reward_function=ReforgedReward(logDir=logDir, GOAL_REWARD=GOAL_REWARD,
                                       GOAL_PUNISHMENT=GOAL_PUNISHMENT, DIST_EXP_FACTOR=DIST_EXP_FACTOR,
                                       TOUCH_REWARD=TOUCH_REWARD,
                                       NULL_PENALTY=NULL_PENALTY, DISTANCE_REWARD_COEF=DISTANCE_REWARD_COEF)
    )


class Learner:
    def __init__(self):
        self.logger = SummaryWriter("log_directory")
        self.algorithm = PPO(actor, critic, self.logger)

        self.buffer = ExperienceBuffer()

        #**DEFAULT NEEDS TO INCORPORATE BASIC SECURITY, THIS IS NOT SUFFICIENT**
        self.redis = Redis(host='127.0.0.1', port=6379)


        # SOREN COMMENT: need to support multiple match args so different envs can do
        # different things
        rl_exe_path = ""
        self.buildWorkers(rl_exe_path, get_match_args)

        # SOREN COMMENT: what's a good model number scheme and do you account for continuing
        # training from a saved model?

        # SOREN COMMENT: this is an ugly way of doing it
        TRAJ_ROUNDS = 10
        traj_count = 0
        while True:
            self.buffer.add_step(self.recieve_worker_data())
            traj_count += 1

            if traj_count >= TRAJ_ROUNDS:
                self.calculate(self.buffer)

                self.buffer.clear()
                traj_count = 0

                # SOREN COMMENT:
                # how do you want to combine the actor/critic dictionary for transmission?
                worker.update_model(self.redit, <<add state dict dump>>, model_version)
                model_version += 1

                # SOREN COMMENT: add in launching extra workers after X amount of time


    # pulled from rolv's wrapper and SB3
    def buildWorkers(self,  match_args_func, path_to_epic_rl, n_envs=1, wait_time=60):
        #def spawn_process():
        #    match = Match(**match_args_func())
        #    env = Gym(match, pipe_id=os.getpid(), path_to_rl=path_to_epic_rl, use_injector=True)
        #    return env

        #env_fns = [spawn_process for _ in range(n_envs)]
        env_fns = [match_args_func for _ in range(n_envs)]

        forkserver_available = "forkserver" in mp.get_all_start_methods()
        start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        for work_remote, remote, env_fn in zip(work_remotes, remotes, env_fns):
            #args = (work_remote, remote, CloudpickleWrapper(env_fn))

            args = (path_to_epic_rl, 1, match_args_func) # ** test this, probably wrong
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



    def calculate(self):
        #apply PPO now but separate so we can refactor to allow different algorithm types
        self.algorithm.calculate(self.buffer)



#this should probably be in its own file
class PPO:
    def __init__(self, actor, critic, logger, n_rollouts = 36, lr_actor = 3e-4, lr_critic = 3e-4, gamma = 0.9, epochs = 1):
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer

        self.logger = logger

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
            # ROLV COMMENT:
            # Is there some overview of different methods [of PPO] (pros/cons)?
            # If there is something that is closer to our task that should be preferred.
            # SOREN COMMENT:
            # I don't know enough to strongly favor one so I'm going with SB3 cause we'll have
            # a reference

            # this is mostly pulled from sb3
            for i, rollout in enumerate(self.buffer.generate_rollouts(self.batch_size)):
                #this should probably be consolidated into buffer
                adv = self.advantages[i:i+batch_size]

                # SOREN COMMENT:
                # need to figure out the right way to do this. agent is a part of worker
                # and agent has the distribution. Return as a part of rollouts?
                <<<<CONTINUE HERE, NEED TO GET THESE>>>>
                log_prob = XXXXX
                old_log_prob = XXXXX
                entropy = XXXXXX

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = adv * ratio
                policy_loss_2 = adv * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # If we want value clipping, add it here
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

