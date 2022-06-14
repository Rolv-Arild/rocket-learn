import cProfile
import io
import os
import pstats
import time
from typing import Iterator

import numba
import numpy as np
import torch
import torch as th
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
        :param batch_size: batch size to break experience data into for training
        :param epochs: Number of epoch when optimizing the loss
        :param minibatch_size: size to break batch sets into (helps combat VRAM issues)
        :param clip_range: PPO Clipping parameter for the value function
        :param ent_coef: Entropy coefficient for the loss calculation
        :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        :param vf_coef: Value function coefficient for the loss calculation
        :param max_grad_norm: optional clip_grad_norm value
        :param logger: wandb logger to store run results
        :param device: torch device
        :param zero_grads_with_none: 0 gradient with None instead of 0

        Look here for info on zero_grads_with_none
        https://pytorch.org/docs/master/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad
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
            zero_grads_with_none=False,
    ):
        self.rollout_generator = rollout_generator

        # TODO let users choose their own agent
        # TODO move agent to rollout generator
        self.agent = agent.to(device)
        self.device = device
        self.zero_grads_with_none = zero_grads_with_none

        self.starting_iteration = 0

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

        self.running_rew_mean = 0
        self.running_rew_var = 1
        self.running_rew_count = 1e-4

        self.total_steps = 0
        self.logger = logger
        self.logger.watch((self.agent.actor, self.agent.critic))
        self.timer = time.time_ns() // 1_000_000
        self.jit_tracer = None

    def update_reward_norm(self, rewards: np.ndarray) -> np.ndarray:
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        batch_count = rewards.shape[0]

        delta = batch_mean - self.running_rew_mean
        tot_count = self.running_rew_count + batch_count

        new_mean = self.running_rew_mean + delta * batch_count / tot_count
        m_a = self.running_rew_var * self.running_rew_count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.running_rew_count * batch_count / (
                self.running_rew_count + batch_count)
        new_var = m_2 / (self.running_rew_count + batch_count)

        new_count = batch_count + self.running_rew_count

        self.running_rew_mean = new_mean
        self.running_rew_var = new_var
        self.running_rew_count = new_count

        return (rewards - self.running_rew_mean) / np.sqrt(self.running_rew_var + 1e-8)  # TODO normalize before update?

    def run(self, iterations_per_save=10, save_dir=None, save_jit=False):
        """
        Generate rollout data and train
        :param iterations_per_save: number of iterations between checkpoint saves
        :param save_dir: where to save
        """
        if save_dir:
            current_run_dir = os.path.join(save_dir, self.logger.project + "_" + str(time.time()))
            os.makedirs(current_run_dir)
        elif iterations_per_save and not save_dir:
            print("Warning: no save directory specified.")
            print("Checkpoints will not be saved.")

        iteration = self.starting_iteration
        rollout_gen = self.rollout_generator.generate_rollouts()

        self.rollout_generator.update_parameters(self.agent.actor)

        while True:
            # pr = cProfile.Profile()
            # pr.enable()
            t0 = time.time()

            def _iter():
                size = 0
                print(f"Collecting rollouts ({iteration})...")
                while size < self.n_steps:
                    try:
                        rollout = next(rollout_gen)
                        if rollout.size() > 0:
                            size += rollout.size()
                            # progress.update(rollout.size())
                            yield rollout
                    except StopIteration:
                        return

            self.calculate(_iter(), iteration)
            iteration += 1

            if save_dir:
                self.save(os.path.join(save_dir, self.logger.project + "_" + "latest"), -1, save_jit)
                if iteration % iterations_per_save == 0:
                    self.save(current_run_dir, iteration, save_jit)  # noqa

            self.rollout_generator.update_parameters(self.agent.actor)

            self.total_steps += self.n_steps  # size
            t1 = time.time()
            self.logger.log({"ppo/steps_per_second": self.n_steps / (t1 - t0), "ppo/total_timesteps": self.total_steps})

            # pr.disable()
            # s = io.StringIO()
            # sortby = pstats.SortKey.CUMULATIVE
            # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            # ps.dump_stats(f"profile_{self.total_steps}")

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

    @staticmethod
    @numba.njit
    def _calculate_advantages_numba(rewards, values, gamma, gae_lambda):
        advantages = np.zeros_like(rewards)
        # v_targets = np.zeros_like(rewards)
        dones = np.zeros_like(rewards)
        dones[-1] = 1.
        episode_starts = np.zeros_like(rewards)
        episode_starts[0] = 1.
        last_values = values[-1]
        last_gae_lam = 0
        size = len(advantages)
        for step in range(size - 1, -1, -1):
            if step == size - 1:
                next_non_terminal = 1.0 - dones[-1].item()
                next_values = last_values
            else:
                next_non_terminal = 1.0 - episode_starts[step + 1].item()
                next_values = values[step + 1]
            v_target = rewards[step] + gamma * next_values * next_non_terminal
            delta = v_target - values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam
            # v_targets[step] = v_target
        return advantages  # , v_targets

    def calculate(self, buffers: Iterator[ExperienceBuffer], iteration):
        """
        Calculate loss and update network
        """
        obs_tensors = []
        act_tensors = []
        # value_tensors = []
        log_prob_tensors = []
        # advantage_tensors = []
        returns_tensors = []

        rewards_tensors = []

        ep_rewards = []
        ep_steps = []
        n = 0

        for buffer in buffers:  # Do discounts for each ExperienceBuffer individually
            if isinstance(buffer.observations[0], (tuple, list)):
                transposed = tuple(zip(*buffer.observations))
                obs_tensor = tuple(torch.from_numpy(np.vstack(t)).float() for t in transposed)
            else:
                obs_tensor = th.from_numpy(np.vstack(buffer.observations)).float()

            with th.no_grad():
                if isinstance(obs_tensor, tuple):
                    try:
                        x = tuple(o.to(self.device) for o in obs_tensor)
                    except RuntimeError as e:
                        print("RuntimeError in obs transfer", e)
                        x = tuple(o.to(self.device) for o in obs_tensor)
                else:
                    try:
                        x = obs_tensor.to(self.device)
                    except RuntimeError as e:
                        print("RuntimeError in obs transfer", e)
                        x = obs_tensor.to(self.device)
                try:
                    values = self.agent.critic(x).detach().cpu().numpy().flatten()  # No batching?
                except RuntimeError as e:
                    print("RuntimeError in critic 1", e)
                    values = self.agent.critic(x).detach().cpu().numpy().flatten()  # No batching?

            actions = np.stack(buffer.actions)
            log_probs = np.stack(buffer.log_probs)
            rewards = np.stack(buffer.rewards)
            dones = np.stack(buffer.dones)

            size = rewards.shape[0]

            episode_starts = np.roll(dones, 1)
            episode_starts[0] = 1.

            advantages = self._calculate_advantages_numba(rewards, values, self.gamma, self.gae_lambda)

            returns = advantages + values

            obs_tensors.append(obs_tensor)
            act_tensors.append(th.from_numpy(actions))
            log_prob_tensors.append(th.from_numpy(log_probs))
            returns_tensors.append(th.from_numpy(returns))
            rewards_tensors.append(th.from_numpy(rewards))

            ep_rewards.append(rewards.sum())
            ep_steps.append(size)
            n += 1
        ep_rewards = np.array(ep_rewards)
        ep_steps = np.array(ep_steps)

        self.logger.log({
            "ppo/ep_reward_mean": ep_rewards.mean(),
            "ppo/ep_reward_std": ep_rewards.std(),
            "ppo/ep_len_mean": ep_steps.mean(),
        }, step=iteration, commit=False)

        if isinstance(obs_tensors[0], tuple):
            transposed = zip(*obs_tensors)
            obs_tensor = tuple(th.cat(t).float() for t in transposed)
        else:
            obs_tensor = th.cat(obs_tensors).float()
        act_tensor = th.cat(act_tensors)
        log_prob_tensor = th.cat(log_prob_tensors).float()
        # advantages_tensor = th.cat(advantage_tensors)
        returns_tensor = th.cat(returns_tensors).float()

        tot_loss = 0
        tot_policy_loss = 0
        tot_entropy_loss = 0
        tot_value_loss = 0
        total_kl_div = 0
        tot_clipped = 0

        n = 0

        if self.jit_tracer is None:
            self.jit_tracer = obs_tensor[0].to(self.device)

        print("Training network...")

        precompute = torch.cat([param.view(-1) for param in self.agent.actor.parameters()])
        t0 = time.perf_counter_ns()
        self.agent.optimizer.zero_grad(set_to_none=self.zero_grads_with_none)
        for e in range(self.epochs):
            # this is mostly pulled from sb3

            indices = torch.randperm(returns_tensor.shape[0])[:self.batch_size]
            if isinstance(obs_tensor, tuple):
                obs_batch = tuple(o[indices] for o in obs_tensor)
            else:
                obs_batch = obs_tensor[indices]
            act_batch = act_tensor[indices]
            log_prob_batch = log_prob_tensor[indices]
            # advantages_batch = advantages_tensor[indices]
            returns_batch = returns_tensor[indices]

            for i in range(0, self.batch_size, self.minibatch_size):
                # Note: Will cut off final few samples

                if isinstance(obs_tensor, tuple):
                    obs = tuple(o[i: i + self.minibatch_size].to(self.device) for o in obs_batch)
                else:
                    obs = obs_batch[i: i + self.minibatch_size].to(self.device)

                act = act_batch[i: i + self.minibatch_size].to(self.device)
                # adv = advantages_batch[i:i + self.minibatch_size].to(self.device)
                ret = returns_batch[i: i + self.minibatch_size].to(self.device)

                old_log_prob = log_prob_batch[i: i + self.minibatch_size].to(self.device)

                # TODO optimization: use forward_actor_critic instead of separate in case shared, also use GPU
                try:
                    log_prob, entropy = self.evaluate_actions(obs, act)  # Assuming obs and actions as input
                except RuntimeError as e:
                    print("RuntimeError in evaluate_actions", e)
                    log_prob, entropy = self.evaluate_actions(obs, act)  # Assuming obs and actions as input

                ratio = torch.exp(log_prob - old_log_prob)

                try:
                    values_pred = self.agent.critic(obs)
                except RuntimeError as e:
                    print("RuntimeError in critic 2", e)
                    values_pred = self.agent.critic(obs)
                except ValueError as e:
                    print("ValueError in evaluate_actions", e)
                    continue

                values_pred = th.squeeze(values_pred)
                adv = ret - values_pred
                adv = (adv - th.mean(adv)) / (th.std(adv) + 1e-8)

                # clipped surrogate loss
                policy_loss_1 = adv * ratio
                policy_loss_2 = adv * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # **If we want value clipping, add it here**
                value_loss = F.mse_loss(ret, values_pred)

                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = entropy

                loss = ((policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss)
                        / (self.batch_size / self.minibatch_size))

                if not torch.isfinite(loss).all():
                    print("Non-finite loss, skipping", n)
                    print("\tPolicy loss:", policy_loss)
                    print("\tEntropy loss:", entropy_loss)
                    print("\tValue loss:", value_loss)
                    print("\tTotal loss:", loss)
                    print("\tRatio:", ratio)
                    print("\tAdv:", adv)
                    print("\tLog prob:", log_prob)
                    print("\tOld log prob:", old_log_prob)
                    print("\tEntropy:", entropy)
                    print("\tActor has inf:", any(not p.isfinite().all() for p in self.agent.actor.parameters()))
                    print("\tCritic has inf:", any(not p.isfinite().all() for p in self.agent.critic.parameters()))
                    print("\tReward as inf:", not np.isfinite(ep_rewards).all())
                    if isinstance(obs, tuple):
                        for j in range(len(obs)):
                            print(f"\tObs[{j}] has inf:", not obs[j].isfinite().all())
                    else:
                        print("\tObs has inf:", not obs.isfinite().all())
                    continue

                loss.backward()

                # Unbiased low variance KL div estimator from http://joschu.net/blog/kl-approx.html
                total_kl_div += th.mean((ratio - 1) - (log_prob - old_log_prob)).item()
                tot_loss += loss.item()
                tot_policy_loss += policy_loss.item()
                tot_entropy_loss += entropy_loss.item()
                tot_value_loss += value_loss.item()
                tot_clipped += th.mean((th.abs(ratio - 1) > self.clip_range).float()).item()
                n += 1
                # pb.update(self.minibatch_size)

            # Clip grad norm
            if self.max_grad_norm is not None:
                clip_grad_norm_(self.agent.actor.parameters(), self.max_grad_norm)

            self.agent.optimizer.step()
            self.agent.optimizer.zero_grad(set_to_none=self.zero_grads_with_none)

        t1 = time.perf_counter_ns()

        assert n > 0

        postcompute = torch.cat([param.view(-1) for param in self.agent.actor.parameters()])
        self.logger.log({
            "ppo/loss": tot_loss / n,
            "ppo/policy_loss": tot_policy_loss / n,
            "ppo/entropy_loss": tot_entropy_loss / n,
            "ppo/value_loss": tot_value_loss / n,
            "ppo/mean_kl": total_kl_div / n,
            "ppo/clip_fraction": tot_clipped / n,
            "ppo/epoch_time": (t1 - t0) / (1e6 * self.epochs),
            "ppo/update_magnitude": th.dist(precompute, postcompute, p=2),
        }, step=iteration, commit=False)  # Is committed after when calculating fps

    def load(self, load_location, continue_iterations=True):
        """
        load the model weights, optimizer values, and metadata
        :param load_location: checkpoint folder to read
        :param continue_iterations: keep the same training steps
        """

        checkpoint = torch.load(load_location)
        self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        # self.agent.shared.load_state_dict(checkpoint['shared_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if continue_iterations:
            self.starting_iteration = checkpoint['epoch']
            self.total_steps = checkpoint["total_steps"]
            print("Continuing training at iteration " + str(self.starting_iteration))

    def save(self, save_location, current_step, save_actor_jit=False):
        """
        Save the model weights, optimizer values, and metadata
        :param save_location: where to save
        :param current_step: the current iteration when saved. Use to later continue training
        """

        version_str = str(self.logger.project) + "_" + str(current_step)
        version_dir = save_location + "\\" + version_str

        os.makedirs(version_dir, exist_ok=current_step == -1)

        torch.save({
            'epoch': current_step,
            "total_steps": self.total_steps,
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            # 'shared_state_dict': self.agent.shared.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            # TODO save/load reward normalization mean, std, count
        }, version_dir + "\\checkpoint.pt")

        if save_actor_jit:
            traced_actor = th.jit.trace(self.agent.actor, self.jit_tracer)
            torch.jit.save(traced_actor, version_dir + "\\jit_policy.jit")
