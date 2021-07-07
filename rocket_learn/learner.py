from typing import Any, List

import cloudpickle
import numpy as np
import torch
import torch as th
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from experience_buffer import ExperienceBuffer
from rocket_learn.rollout_generator.base_rollout_generator import BaseRolloutGenerator
from simple_agents import PPOAgent


class CloudpickleWrapper:
    """
    ** Copied from SB3 **

    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

    :param var: the variable you wish to wrap for pickling with cloudpickle
    """

    def __init__(self, var: Any):
        self.var = var

    def __getstate__(self) -> Any:
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var: Any) -> None:
        self.var = cloudpickle.loads(var)


# this should probably be in its own file
class PPO:
    def __init__(self, rollout_generator: BaseRolloutGenerator, actor, critic, lr_actor=3e-4, lr_critic=3e-4, gamma=0.9, epochs=1):
        self.rollout_generator = rollout_generator
        self.agent = PPOAgent(actor, critic)  # TODO let users choose their own agent

        # hyperparameters
        self.epochs = epochs
        self.gamma = gamma
        self.n_rollouts = 36
        self.lmbda = 1.
        self.gae_lambda = 0
        self.batch_size = 512
        self.clip_range = .2
        self.ent_coef = 1
        self.vf_coef = 1
        self.optimizer = torch.optim.Adam([
            {'params': self.agent.actor.parameters(), 'lr': lr_actor},
            {'params': self.agent.critic.parameters(), 'lr': lr_critic}
        ])

    def run(self):
        while True:
            rollout_gen = self.rollout_generator.generate_rollouts()

            # SOREN COMMENT: this is an ugly way of doing it
            while True:
                rollouts = []
                while len(rollouts) < self.n_rollouts:
                    try:
                        rollout = next(rollout_gen)
                        rollouts.append(rollout)
                    except StopIteration:
                        return

                for rollout in rollouts:
                    self.calculate(rollout)

                self.rollout_generator.update_parameters(self.agent)

    def set_logger(self, logger):
        self.logger = logger

    def policy_dict_list(self):
        networks = [self.agent.actor.get_state_dict(), self.agent.critic.get_state_dict()]
        return networks

    def evaluate_actions(self):
        ## **TODO**
        return -1, -1

    def calculate(self, buffer: ExperienceBuffer):
        # buffer = th.zeros((self.batch_size, 67))  # TODO extract and shuffle from input
        print("Opt step")
        return
        values = self.agent.critic(buffer)
        buffer_size = buffer.size()

        # totally stole this section from
        # https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8
        # I am not attached to it, make it better if you'd like
        returns = []
        gae = 0
        for i in reversed(range(buffer_size)):
            delta = buffer.rewards[i] + self.gamma * values[i + 1] * buffer.dones[i] - values[i]
            gae = delta + self.gamma * self.lmbda * buffer.dones[i] * gae
            returns.insert(0, gae + values[i])

        advantages = np.array(returns) - values[:-1]
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
        # returns is also called references?

        for e in range(self.epochs):
            # SOREN COMMENT:
            # I'm going with SB3 PPO implementation cause we'll have a reference

            # this is mostly pulled from sb3
            for i, rollout in enumerate(buffer.generate_slices(self.batch_size)):
                print("Optimizer step", i)
                continue
                adv = advantages[i:i + self.batch_size]

                log_prob, entropy = self.evaluate_actions()
                old_log_prob = rollout.log_prob
                ratio = torch.exp(log_prob - old_log_prob)

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

                # self.logger write here to log results
