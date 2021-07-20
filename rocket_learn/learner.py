from typing import Any, List

import cloudpickle
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.rollout_generator.base_rollout_generator import BaseRolloutGenerator
from rocket_learn.simple_agents import PPOAgent


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
    def __init__(self, rollout_generator: BaseRolloutGenerator, actor, critic, n_rollouts=36, lr_actor=3e-4,
                 lr_critic=3e-4, gamma=0.9, batch_size=512, epochs=1):
        self.rollout_generator = rollout_generator
        self.agent = PPOAgent(actor, critic)  # TODO let users choose their own agent

        # hyperparameters
        self.epochs = epochs
        self.gamma = gamma
        self.n_rollouts = n_rollouts
        self.lmbda = 1.
        self.gae_lambda = 0
        self.batch_size = batch_size
        self.clip_range = .2
        self.ent_coef = 1
        self.vf_coef = 1
        self.max_grad_norm = None
        self.optimizer = torch.optim.Adam([
            {'params': self.agent.actor.parameters(), 'lr': lr_actor},
            {'params': self.agent.critic.parameters(), 'lr': lr_critic}
        ])

    def run(self):
        while True:
            rollout_gen = self.rollout_generator.generate_rollouts()

            while True:
                rollouts = []
                while len(rollouts) < self.n_rollouts:
                    try:
                        rollout = next(rollout_gen)
                        rollouts.append(rollout)
                    except StopIteration:
                        return

                self.calculate(rollouts)

                self.rollout_generator.update_parameters([self.agent.actor.state_dict(),
                                                          self.agent.critic.state_dict()])

    def set_logger(self, logger):
        self.logger = logger

    def policy_dict_list(self):
        networks = [self.agent.actor.get_state_dict(), self.agent.critic.get_state_dict()]
        return networks

    def evaluate_actions(self, observations, actions):
        dists = self.agent.get_action_distribution(observations)
        indices = self.agent.get_action_indices(dists)

        # Thanks Rangler!
        new_raw_a_logits = self.agent.forward_actor(observations)

        new_raw_a_probs = [F.softmax(a_logit, dim=-1) for a_logit in new_raw_a_logits]

        new_cat_a_probs = th.cat(
            (new_raw_a_probs[0], new_raw_a_probs[1], new_raw_a_probs[2], new_raw_a_probs[3], new_raw_a_probs[4]), 1)
        new_cat_probs = new_cat_a_probs.gather(1, actions[:, :5])

        new_ber_a_probs = th.cat((new_raw_a_probs[5], new_raw_a_probs[6], new_raw_a_probs[7]), 1)
        new_ber_probs = new_ber_a_probs.gather(1, actions[:, 5:])

        log_prob = torch.cat([new_cat_probs, new_ber_probs], dim=1)
        log_prob = log_prob.sum(dim=1)

        cat_entropy = torch.sum(new_cat_a_probs * torch.log(new_cat_a_probs + 1e-10), dim=(0, 1))
        ber_entropy = torch.sum(new_ber_a_probs * torch.log(new_ber_a_probs + 1e-10), dim=(0, 1))
        entropy = -torch.mean(cat_entropy + ber_entropy)

        return log_prob, entropy

    def calculate(self, buffers: List[ExperienceBuffer]):
        obs_tensors = []
        act_tensors = []
        log_prob_tensors = []
        rew_tensors = []
        done_tensors = []

        for buffer in buffers:
            obs_tensor = th.as_tensor(np.stack(buffer.observations))
            act_tensor = th.as_tensor(np.stack(buffer.actions))
            log_prob_tensor = th.as_tensor(buffer.log_prob)
            rew_tensor = th.as_tensor(buffer.rewards)  # TODO discounted rewards (returns? not in python preferably)
            done_tensor = th.as_tensor(buffer.dones)

            obs_tensors.append(obs_tensor)
            act_tensors.append(act_tensor)
            log_prob_tensors.append(log_prob_tensor)
            rew_tensors.append(rew_tensor)
            done_tensors.append(done_tensor)

        # Set device?
        # THIS NEEDS TO HAPPEN AFTER ADV IS CALCULATED
        obs_tensor = th.cat(obs_tensors).float()
        act_tensor = th.cat(act_tensors)
        log_prob_tensor = th.cat(log_prob_tensors).float()
        rew_tensor = th.cat(rew_tensors).float()
        done_tensor = th.cat(done_tensors)

        # **sb3 version if we want it**
        size = rew_tensor.size()[0]
        advantages = np.zeros((size), dtype=np.float32)
        v_targets = np.zeros((size), dtype=np.float32)

        # dones shifted right by 1 and then 1 at the very beginning
        episode_starts = np.roll(done_tensor, 1)
        episode_starts[0] = 1
        episode_starts = th.as_tensor(episode_starts)

        #with th.no_grad():
        values = self.agent.forward_critic(obs_tensor).detach().cpu().numpy().flatten()
        last_values = values[-1]
        last_gae_lam = 0
        for step in reversed(range(size)):
            if step == size - 1:
                next_non_terminal = 1.0 - done_tensor[-1].item()
                next_values = last_values
            else:
                next_non_terminal = 1.0 - episode_starts[step + 1].item()
                next_values = values[step + 1]
            v_target = rew_tensor[step] + self.gamma * next_values
            delta = v_target * next_non_terminal - values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam
            v_targets[step] = v_target
        returns = advantages + values

        v_targets = th.as_tensor(v_targets)
        returns = th.as_tensor(returns)
        advantages = th.as_tensor(advantages)
        advantages = (advantages - th.mean(advantages)) / (th.std(advantages) + 1e-8)
        #advantages.detach_()



        # shuffle data
        indices = torch.randperm(advantages.shape[0])
        obs_tensor = obs_tensor[indices]
        act_tensor = act_tensor[indices]
        log_prob_tensor = log_prob_tensor[indices]
        advantages = advantages[indices]
        returns = returns[indices]

        for e in range(self.epochs):
            # this is mostly pulled from sb3
            for i in range(0, obs_tensor.shape[0], self.batch_size):
                # Note: Will cut off final few samples

                obs = obs_tensor[i: i + self.batch_size]
                act = act_tensor[i: i + self.batch_size]
                adv = advantages[i:i + self.batch_size]
                rew = returns[i: i + self.batch_size]

                old_log_prob = log_prob_tensor[i: i + self.batch_size]

                log_prob, entropy = self.evaluate_actions(obs, act)  # Assuming obs and actions as input
                ratio = torch.exp(log_prob - old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = adv * ratio
                policy_loss_2 = adv * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # **If we want value clipping, add it here**
                val = self.agent.forward_critic(obs)[1:]
                target = rew[:-1] + self.gamma * val
                value_loss = F.mse_loss(target, val)

                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = entropy

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with torch.no_grad():
                    log_ratio = log_prob - old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()

                self.optimizer.zero_grad()

                loss.backward()
                
                # Clip grad norm
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # self.logger write here to log results
