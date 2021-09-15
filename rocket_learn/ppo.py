import io
import time
from typing import Optional, Type, Iterator
import os

import numpy as np
import torch
import torch as th
import tqdm
from torch import nn
from torch.nn import functional as F, Identity

from rocket_learn.agent import BaseAgent
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.rollout_generator.base_rollout_generator import BaseRolloutGenerator


def _default_collate(observations):
    return torch.as_tensor(np.stack(observations)).float()


class PPOAgent(BaseAgent):
    """
    Agent designed to work with PPO
    """
    def __init__(self, actor: nn.Module, critic: nn.Module, shared: Optional[nn.Module] = None, collate_fn=None):
        super().__init__()
        self.actor = actor
        self.critic = critic
        if shared is None:
            shared = Identity()
        self.shared = shared
        self.collate_fn = _default_collate if collate_fn is None else collate_fn

    def to(self, device):
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        if self.shared is not None:
            self.shared = self.shared.to(device)

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

    def get_model_params(self):
        buf = io.BytesIO()
        torch.save([self.actor, self.critic, self.shared], buf)
        return buf

    def set_model_params(self, params) -> None:
        torch.load(params.read())


class PPO:
    """
        Proximal Policy Optimization algorithm (PPO)

        :param rollout_generator: Function that will generate the rollouts
        :param actor: Torch actor network
        :param critic: Torch critic network
        :param n_steps: The number of steps to run per update
        :param lr_actor: Actor optimizer learning rate (Adam)
        :param lr_critic: Critic optimizer learning rate (Adam)
        :param lr_shared: Shared layer optimizer learning rate (Adam)
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
            agent: PPOAgent,
            n_steps=4096,
            lr_actor=3e-4,
            lr_critic=3e-4,
            lr_shared=3e-4,
            gamma=0.99,
            batch_size=512,
            epochs=10,
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
        self.agent = agent
        self.agent.to(device)
        self.device = device

        self.starting_epoch = 0


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

        self.logger = logger
        self.logger.watch((self.agent.actor, self.agent.critic, self.agent.shared))

        self.optimizer = optimizer_class([
            {'params': self.agent.actor.parameters(), 'lr': lr_actor},
            {'params': self.agent.critic.parameters(), 'lr': lr_critic},
            {'params': self.agent.shared.parameters(), 'lr': lr_shared}
        ])

    def run(self, epochs_per_save=None, save_dir=None):
        """
        Generate rollout data and train
        :param epochs_per_save: number of epoches between checkpoint saves
        :param save_dir: where to save
        """
        if save_dir:
            current_run_dir = save_dir+"\\"+self.logger.project+"_"+str(time.time())
            os.makedirs(current_run_dir)
        elif epochs_per_save and not save_dir:
            print("Warning: no save directory specified.")
            print("Checkpoints will not be save.")

        epoch = self.starting_epoch
        rollout_gen = self.rollout_generator.generate_rollouts()

        while True:
            t0 = time.time()
            self.rollout_generator.update_parameters(self.agent)

            def _iter():
                size = 0
                progress = tqdm.tqdm(desc=f"PPO_iter_{epoch}", total=self.n_steps)
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
                self.save(current_run_dir, epoch)

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
            obs_tensor = self.agent.collate_fn(buffer.observations)
            # obs_tensor = th.as_tensor(np.stack(buffer.observations)).float()
            act_tensor = th.as_tensor(np.stack(buffer.actions))
            log_prob_tensor = th.as_tensor(np.stack(buffer.log_prob))
            rew_tensor = th.as_tensor(np.stack(buffer.rewards))
            done_tensor = th.as_tensor(np.stack(buffer.dones))

            log_prob_tensor.detach_()

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
                values = self.agent.forward_critic(x).detach().cpu().numpy().flatten()  # No batching?
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
        indices = torch.randperm(advantages_tensor.shape[0])[:self.n_steps]
        if isinstance(obs_tensor, tuple):
            obs_tensor = tuple(o[indices] for o in obs_tensor)
        else:
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

                if isinstance(obs_tensor, tuple):
                    obs = tuple(o[i: i + self.batch_size].to(self.device) for o in obs_tensor)
                else:
                    obs = obs_tensor[i: i + self.batch_size].to(self.device)

                act = act_tensor[i: i + self.batch_size].to(self.device)
                adv = advantages[i:i + self.batch_size].to(self.device)
                ret = returns[i: i + self.batch_size].to(self.device)

                old_log_prob = log_prob_tensor[i: i + self.batch_size].to(self.device)

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
                tot_loss += loss.item()
                tot_policy_loss += policy_loss.item()
                tot_entropy_loss += entropy_loss.item()
                tot_value_loss += value_loss.item()
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

    def load(self, load_location):
        """
        load the model weights, optimizer values, and metadata
        :param load_location: checkpoint folder to read
        :return:
        """

        checkpoint = torch.load(load_location)
        self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.agent.shared.load_state_dict(checkpoint['shared_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
            'shared_state_dict': self.agent.shared.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, version_dir + "\\checkpoint.pt")

