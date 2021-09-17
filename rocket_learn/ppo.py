import cProfile
import os
import pstats
import time
from pstats import SortKey
from typing import Type, Iterator, Union, Iterable

import numpy as np
import torch
import torch as th
import tqdm
from torch import nn
from torch._six import inf
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.rollout_generator.base_rollout_generator import BaseRolloutGenerator


class PPO:
    """
        Proximal Policy Optimization algorithm (PPO)

        :param rollout_generator: Function that will generate the rollouts
        :param agent: An ActorCriticAgent
        :param n_steps: The number of steps to run per update
        :param gamma: Discount factor
        :param batch_size: Minibatch size
        :param epochs: Number of epoch when optimizing the loss
        :param clip_range: Clipping parameter for the value function
        :param ent_coef: Entropy coefficient for the loss calculation
        :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        :param vf_coef: Value function coefficient for the loss calculation
    """

    def __init__(
            self,
            rollout_generator: BaseRolloutGenerator,
            agent: ActorCriticAgent,
            n_steps=4096,
            gamma=0.99,
            batch_size=512,
            epochs=10,
            # reuse=2,
            minibatch_size=None,
            clip_range=0.2,
            ent_coef=0.01,
            gae_lambda=0.95,
            vf_coef=1,
            max_grad_norm=0.5,
            logger=None,
            device="cuda",
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam
    ):
        self.rollout_generator = rollout_generator

        # TODO let users choose their own agent
        # TODO move agent to rollout generator
        self.agent = agent.to(device)
        self.device = device

        self.starting_epoch = 0

        # hyperparameters
        self.epochs = epochs
        self.gamma = gamma
        # assert n_steps % batch_size == 0
        # self.reuse = reuse
        self.n_steps = n_steps
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size or batch_size
        assert self.batch_size % self.minibatch_size == 0
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.logger = logger
        self.logger.watch((self.agent.actor, self.agent.critic))

    def run(self, epochs_per_save=10, save_dir=None):
        """
        Generate rollout data and train
        :param epochs_per_save: number of epochs between checkpoint saves
        :param save_dir: where to save
        """
        if save_dir:
            current_run_dir = os.path.join(save_dir, self.logger.project + "_" + str(time.time()))
            os.makedirs(current_run_dir)
        elif epochs_per_save and not save_dir:
            print("Warning: no save directory specified.")
            print("Checkpoints will not be saved.")

        epoch = self.starting_epoch
        rollout_gen = self.rollout_generator.generate_rollouts()

        while True:
            t0 = time.time()
            self.rollout_generator.update_parameters(self.agent.actor)

            def _iter():
                size = 0
                progress = tqdm.tqdm(desc=f"Collecting rollouts ({epoch})", total=self.n_steps, position=0, leave=True)
                while size < self.n_steps:
                    try:
                        rollout = next(rollout_gen)
                        size += rollout.size()
                        progress.update(rollout.size())
                        yield rollout
                    except StopIteration:
                        return

            self.calculate(_iter())
            epoch += 1
            t1 = time.time()
            self.logger.log({"fps": self.n_steps / (t1 - t0)})

            if save_dir and epoch % epochs_per_save == 0:
                self.save(current_run_dir, epoch)  # noqa

    def set_logger(self, logger):
        self.logger = logger

    def evaluate_actions(self, observations, actions):
        """
        Calculate Log Probability and Entropy of actions
        """
        dist = self.agent.actor.get_action_distribution(observations)
        # indices = self.agent.get_action_indices(dists)

        log_prob = self.agent.actor.log_prob(dist, actions)
        entropy = self.agent.actor.entropy(dist, actions)

        entropy = -torch.mean(entropy)
        return log_prob, entropy

    def calculate(self, buffers: Iterator[ExperienceBuffer]):
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

        ep_rewards = []
        ep_steps = []
        n = 0

        for buffer in buffers:  # Do discounts for each ExperienceBuffer individually
            if isinstance(buffer.observations[0], (tuple, list)):
                transposed = tuple(zip(*buffer.observations))
                obs_tensor = tuple(torch.as_tensor(np.vstack(t)).float() for t in transposed)
            else:
                obs_tensor = th.as_tensor(np.stack(buffer.observations)).float()
            # obs_tensor = th.as_tensor(np.stack(buffer.observations)).float()
            act_tensor = th.as_tensor(np.stack(buffer.actions))
            log_prob_tensor = th.as_tensor(np.stack(buffer.log_prob))
            rew_tensor = th.as_tensor(np.stack(buffer.rewards))
            done_tensor = th.as_tensor(np.stack(buffer.dones))

            log_prob_tensor.detach_()  # Unnecessary?

            size = rew_tensor.size()[0]
            advantages = th.zeros((size,), dtype=th.float)
            v_targets = th.zeros((size,), dtype=th.float)

            episode_starts = th.roll(done_tensor, 1)
            episode_starts[0] = 1.

            with th.no_grad():
                if isinstance(obs_tensor, tuple):
                    x = tuple(o.to(self.device) for o in obs_tensor)
                else:
                    x = obs_tensor.to(self.device)
                values = self.agent.critic(x).detach().cpu().numpy().flatten()  # No batching?
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

            ep_rewards.append(rew_tensor.sum())
            ep_steps.append(size)
            n += 1
        ep_rewards = np.array(ep_rewards)
        ep_steps = np.array(ep_steps)

        self.logger.log({
            "ep_reward_mean": ep_rewards.mean(),
            "ep_reward_std": ep_rewards.std(),
            "ep_len_mean": ep_steps.mean(),
        }, commit=False)

        if isinstance(obs_tensors[0], tuple):
            transposed = zip(*obs_tensors)
            obs_tensor = tuple(th.cat(t).float() for t in transposed)
        else:
            obs_tensor = th.cat(obs_tensors).float()
        act_tensor = th.cat(act_tensors)
        log_prob_tensor = th.cat(log_prob_tensors).float()
        advantages_tensor = th.cat(advantage_tensors)
        returns_tensor = th.cat(returns_tensors)

        # rewards_tensor = th.cat(rewards_tensors)
        # print(th.mean(rewards_tensor).item())

        # shuffle data
        # indices = torch.randperm(advantages_tensor.shape[0])[:self.n_steps]
        # if isinstance(obs_tensor, tuple):
        #     obs_tensor = tuple(o[indices] for o in obs_tensor)
        # else:
        #     obs_tensor = obs_tensor[indices]
        # act_tensor = act_tensor[indices]
        # log_prob_tensor = log_prob_tensor[indices]
        # advantages = advantages_tensor[indices]
        # returns = returns_tensor[indices]

        tot_loss = 0
        tot_policy_loss = 0
        tot_entropy_loss = 0
        tot_value_loss = 0
        n = 0

        pr = cProfile.Profile()
        pr.enable()

        pb = tqdm.tqdm(desc="Training network", total=self.epochs * self.batch_size, position=0, leave=True)

        self.agent.optimizer.zero_grad()
        for e in range(self.epochs):
            # this is mostly pulled from sb3

            indices = torch.randperm(advantages_tensor.shape[0])[:self.batch_size]
            if isinstance(obs_tensor, tuple):
                obs_batch = tuple(o[indices] for o in obs_tensor)
            else:
                obs_batch = obs_tensor[indices]
            act_batch = act_tensor[indices]
            log_prob_batch = log_prob_tensor[indices]
            advantages_batch = advantages_tensor[indices]
            returns_batch = returns_tensor[indices]

            for i in range(0, self.batch_size, self.minibatch_size):
                # Note: Will cut off final few samples

                if isinstance(obs_tensor, tuple):
                    obs = tuple(o[i: i + self.minibatch_size].to(self.device) for o in obs_batch)
                else:
                    obs = obs_batch[i: i + self.minibatch_size].to(self.device)

                act = act_batch[i: i + self.minibatch_size].to(self.device)
                adv = advantages_batch[i:i + self.minibatch_size].to(self.device)
                ret = returns_batch[i: i + self.minibatch_size].to(self.device)

                old_log_prob = log_prob_batch[i: i + self.minibatch_size].to(self.device)

                # TODO optimization: use forward_actor_critic instead of separate in case shared, also use GPU
                log_prob, entropy = self.evaluate_actions(obs, act)  # Assuming obs and actions as input
                ratio = torch.exp(log_prob - old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = adv * ratio
                policy_loss_2 = adv * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # **If we want value clipping, add it here**
                values_pred = self.agent.critic(obs)
                values_pred = th.squeeze(values_pred)
                value_loss = F.mse_loss(ret, values_pred)

                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = entropy

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                if not torch.isfinite(loss).all():
                    print("And I oop")

                loss.backward()

                # *** self.logger write here to log results ***
                tot_loss += loss.item()
                tot_policy_loss += policy_loss.item()
                tot_entropy_loss += entropy_loss.item()
                tot_value_loss += value_loss.item()
                n += 1
                pb.update(self.minibatch_size)

            # Clip grad norm
            if self.max_grad_norm is not None:
                clip_grad_norm_(self.agent.actor.parameters(), self.max_grad_norm)

            self.agent.optimizer.step()
            self.agent.optimizer.zero_grad()

        pr.disable()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr).sort_stats(sortby).dump_stats("policy_update.pstats")
        print("Hei")

        self.logger.log({
            "loss": tot_loss / n,
            "policy_loss": tot_policy_loss / n,
            "entropy_loss": tot_entropy_loss / n,
            "value_loss": tot_value_loss / n,
        }, commit=False)  # Is committed after when calculating fps

    def load(self, load_location):
        """
        load the model weights, optimizer values, and metadata
        :param load_location: checkpoint folder to read
        :return:
        """

        checkpoint = torch.load(load_location)
        self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        # self.agent.shared.load_state_dict(checkpoint['shared_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.starting_epoch = checkpoint['epoch']

        print("Continuing training at epoch " + str(self.starting_epoch))

    def save(self, save_location, current_step):
        """
        Save the model weights, optimizer values, and metadata
        :param save_location: where to save
        :param epoch: the current epoch when saved. Use to later continue training
        """

        version_str = str(self.logger.project) + "_" + str(current_step)
        version_dir = save_location + "\\" + version_str

        os.makedirs(version_dir)

        torch.save({
            'epoch': current_step,
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            # 'shared_state_dict': self.agent.shared.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
        }, version_dir + "\\checkpoint.pt")
