import io
import os
import time
from typing import List, Optional

import numpy as np
import torch
import torch as th
from torch import nn
from torch.nn import functional as F, Identity

from rocket_learn.agent import BaseAgent
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.rollout_generator.base_rollout_generator import BaseRolloutGenerator

import wandb


class PPOAgent(BaseAgent):
    def __init__(self, actor: nn.Module, critic: nn.Module, shared: Optional[nn.Module] = None):
        super().__init__()
        self.actor = actor
        self.critic = critic
        if shared is None:
            shared = Identity()
        self.shared = shared

    def forward_actor_critic(self, obs):
        if self.shared is not None:
            obs = self.shared(obs)
        return self.actor(obs), self.critic(obs)

    def forward_actor(self, obs):
        if self.shared is not None:
            obs = self.shared(obs)
        return self.actor(obs)

    def forward_critic(self, obs):
        if self.shared is not None:
            obs = self.shared(obs)
        return self.critic(obs)

    def serialize(self):
        buf = io.BytesIO()
        torch.save([self.actor, self.critic, self.shared], buf)
        return buf

    def get_model_params(self, params):
        return self.actor.state_dict(), self.critic.state_dict()

    def set_model_params(self, params) -> None:
        self.actor.load_state_dict(params[0])
        self.critic.load_state_dict(params[1])


class PPO:
    """
        Proximal Policy Optimization algorithm (PPO)

        :param rollout_generator: Function that will generate the rollouts
        :param actor: Torch actor network
        :param critic: Torch critic network
        :param n_steps: The number of steps to run per update
        :param lr_actor: Actor optimizer learning rate (Adam)
        :param lr_critic: Critic optimizer learning rate (Adam)
        :param gamma: Discount factor
        :param batch_size: Minibatch size
        :param epochs: Number of epoch when optimizing the loss
        :param clip_range: Clipping parameter for the value function
        :param ent_coef: Entropy coefficient for the loss calculation
        :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        :param vf_coef: Value function coefficient for the loss calculation
    """

    def __init__(self, rollout_generator: BaseRolloutGenerator, agent: PPOAgent, n_steps=4096, lr_actor=3e-4,
                 lr_critic=3e-4, lr_shared=3e-4, gamma=0.99, batch_size=512, epochs=10, clip_range=0.2, ent_coef=0.01,
                 gae_lambda=0.95, vf_coef=1, max_grad_norm=0.5, logger=None):
        self.rollout_generator = rollout_generator

        # TODO let users choose their own agent
        # TODO move agent to rollout generator
        self.agent = agent

        self.logger = logger

        # hyperparameters
        self.epochs = epochs
        self.gamma = gamma
        assert n_steps % batch_size == 0
        self.n_steps = n_steps
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.logger.watch([self.agent.actor, self.agent.critic, self.agent.shared])

        # **TODO** allow different optimizer types
        self.optimizer = torch.optim.Adam([
            {'params': self.agent.actor.parameters(), 'lr': lr_actor},
            {'params': self.agent.critic.parameters(), 'lr': lr_critic},
            {'params': self.agent.shared.parameters(), 'lr': lr_shared}
        ])

    def run(self):
        """
        Generate rollout data and train
        """
        epoch = 0
        rollout_gen = self.rollout_generator.generate_rollouts()

        while True:
            t0 = time.time()
            self.rollout_generator.update_parameters(self.agent)

            rollouts = []
            size = 0
            while size < self.n_steps:
                try:
                    rollout = next(rollout_gen)
                    rollouts.append(rollout)
                    size += rollout.size()
                except StopIteration:
                    return

            self.calculate(rollouts)
            epoch += 1
            t1 = time.time()
            self.logger.log({"fps": size / (t1 - t0)})

    def set_logger(self, logger):
        self.logger = logger

    def evaluate_actions(self, observations, actions):
        """
        Calculate Log Probability and Entropy of actions
        """
        dists = self.agent.get_action_distribution(observations)
        # indices = self.agent.get_action_indices(dists)

        log_prob = th.stack(
            [dist.log_prob(action) for dist, action in zip(dists, th.unbind(actions, dim=1))], dim=1
        ).sum(dim=1)

        entropy = th.stack([dist.entropy() for dist in dists], dim=1).sum(dim=1)
        entropy = -torch.mean(entropy)
        return log_prob, entropy

    def calculate(self, buffers: List[ExperienceBuffer]):
        """
        Calculate loss and update network
        """
        obs_tensors = []
        act_tensors = []
        # value_tensors = []
        log_prob_tensors = []
        advantage_tensors = []
        returns_tensors = []
        v_target_tensors = []

        rewards_tensors = []

        ep_rewards = np.zeros(len(buffers))
        ep_steps = np.zeros(len(buffers))
        n = 0
        for buffer in buffers:  # Do discounts for each ExperienceBuffer individually
            obs_tensor = th.as_tensor(np.stack(buffer.observations)).float()
            act_tensor = th.as_tensor(np.stack(buffer.actions))
            log_prob_tensor = th.as_tensor(buffer.log_prob)
            rew_tensor = th.as_tensor(buffer.rewards)
            done_tensor = th.as_tensor(buffer.dones)

            log_prob_tensor.detach_()

            size = rew_tensor.size()[0]
            advantages = th.zeros((size,), dtype=th.float)
            v_targets = th.zeros((size,), dtype=th.float)

            episode_starts = th.roll(done_tensor, 1)
            episode_starts[0] = 1.

            with th.no_grad():
                values = self.agent.forward_critic(obs_tensor).detach().cpu().numpy().flatten()  # No batching?
                last_values = values[-1]
                last_gae_lam = 0
                for step in reversed(range(size)):
                    if step == size - 1:
                        next_non_terminal = 1.0 - done_tensor[-1].item()
                        next_values = last_values
                    else:
                        next_non_terminal = 1.0 - episode_starts[step + 1].item()
                        next_values = values[step + 1]
                    v_target = rew_tensor[step] + self.gamma * next_values * next_non_terminal
                    delta = v_target - values[step]
                    last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                    advantages[step] = last_gae_lam
                    v_targets[step] = v_target

            returns = advantages + values
            advantages = (advantages - th.mean(advantages)) / (th.std(advantages) + 1e-8)

            obs_tensors.append(obs_tensor)
            act_tensors.append(act_tensor)
            log_prob_tensors.append(log_prob_tensor)
            advantage_tensors.append(advantages)
            returns_tensors.append(returns)
            v_target_tensors.append(v_targets)
            rewards_tensors.append(rew_tensor)

            ep_rewards[n] = rew_tensor.sum()
            ep_steps[n] = size
            n += 1
        self.logger.log({
            "ep_reward_mean": ep_rewards.mean(),
            "ep_reward_std": ep_rewards.std(),
            "ep_len_mean": ep_steps.mean(),
        }, commit=False)

        obs_tensor = th.cat(obs_tensors).float()
        act_tensor = th.cat(act_tensors)
        log_prob_tensor = th.cat(log_prob_tensors).float()
        advantages_tensor = th.cat(advantage_tensors)
        returns_tensor = th.cat(returns_tensors)

        # rewards_tensor = th.cat(rewards_tensors)
        # print(th.mean(rewards_tensor).item())

        # shuffle data
        indices = torch.randperm(advantages_tensor.shape[0])[:self.n_steps]
        obs_tensor = obs_tensor[indices]
        act_tensor = act_tensor[indices]
        log_prob_tensor = log_prob_tensor[indices]
        advantages = advantages_tensor[indices]
        returns = returns_tensor[indices]

        tot_loss = 0
        tot_policy_loss = 0
        tot_entropy_loss = 0
        tot_value_loss = 0
        n = 0

        for e in range(self.epochs):
            # this is mostly pulled from sb3
            for i in range(0, self.n_steps, self.batch_size):
                # Note: Will cut off final few samples

                obs = obs_tensor[i: i + self.batch_size]
                act = act_tensor[i: i + self.batch_size]
                adv = advantages[i:i + self.batch_size]
                ret = returns[i: i + self.batch_size]

                old_log_prob = log_prob_tensor[i: i + self.batch_size]

                # TODO optimization: use forward_actor_critic instead of separate in case shared, also use GPU
                log_prob, entropy = self.evaluate_actions(obs, act)  # Assuming obs and actions as input
                ratio = torch.exp(log_prob - old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = adv * ratio
                policy_loss_2 = adv * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # **If we want value clipping, add it here**
                values_pred = self.agent.forward_critic(obs)
                values_pred = th.squeeze(values_pred)
                value_loss = F.mse_loss(ret, values_pred)

                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = entropy

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()

                # Clip grad norm
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # *** self.logger write here to log results ***
                tot_loss += loss
                tot_policy_loss += policy_loss
                tot_entropy_loss += entropy_loss
                tot_value_loss += value_loss
                n += 1

                # loss
                # policy loss
                # entropy loss
                # value loss

                # average rewards
                # average episode length

                # hyper parameter logging or JSON info
        self.logger.log(
            {
                "loss": tot_loss / n,
                "policy_loss": tot_policy_loss / n,
                "entropy_loss": tot_entropy_loss / n,
                "value_loss": tot_value_loss / n,
            },
        )
