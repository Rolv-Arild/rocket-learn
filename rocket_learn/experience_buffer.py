import torch as th
import numpy as np

class ExperienceBuffer:
    def __init__(self, meta=None, observations=None, actions=None, rewards=None, dones=None,
                 log_probs=None, infos=None, returns=None):
        self.meta = meta
        self.result = 0
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.infos = []
        self.returns = []

        if observations is not None:
            self.observations = observations

        if actions is not None:
            self.actions = actions

        if rewards is not None:
            self.rewards = rewards

        if dones is not None:
            self.dones = dones  # TODO Done probably doesn't need to be a list, will always just be false until last?

        if log_probs is not None:
            self.log_probs = log_probs

        if infos is not None:
            self.infos = infos

        if returns is not None:
            self.returns = returns

    def size(self):
        return len(self.dones)

    def add_step(self, observation, action, reward, done, log_prob, info):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.infos.append(info)

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.infos = []
        self.returns = []

    def generate_slices(self, batch_size):
        for i in range(0, len(self.observations), batch_size):
            yield ExperienceBuffer(self.observations[i:i+batch_size],
                                   self.actions[i:i+batch_size],
                                   self.rewards[i:i+batch_size],
                                   self.dones[i:i+batch_size],
                                   self.log_probs[i:i+batch_size],
                                   self.infos[i:i+batch_size],
                                   self.returns[i:i+batch_size])


    def calculate_returns(self, critic, gamma, gae_lambda, device):
        assert self.size() is not 0

        rew_tensor = th.as_tensor(np.stack(self.rewards))
        done_tensor = th.as_tensor(np.stack(self.dones))
        obs_tensor = th.as_tensor(np.stack(self.observations))

        size = rew_tensor.size()[0]
        advantages = th.zeros((size,), dtype=th.float)
        v_targets = th.zeros((size,), dtype=th.float)

        episode_starts = th.roll(done_tensor, 1)
        episode_starts[0] = 1.

        with th.no_grad():
            if isinstance(obs_tensor, tuple):
                x = tuple(o.to(device) for o in obs_tensor)
            else:
                x = obs_tensor.to(device)
            x = x.float()
            values = critic(x).detach().cpu().numpy().flatten()  # No batching?

            last_values = values[-1]
            last_gae_lam = 0
            for step in reversed(range(size)):
                if step == size - 1:
                    next_non_terminal = 1.0 - done_tensor[-1].item()
                    next_values = last_values
                else:
                    next_non_terminal = 1.0 - episode_starts[step + 1].item()
                    next_values = values[step + 1]
                v_target = rew_tensor[step] + gamma * next_values * next_non_terminal
                delta = v_target - values[step]
                last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
                advantages[step] = last_gae_lam
                v_targets[step] = v_target

        returns = advantages + values
        self.returns = returns
        assert self.returns == self.size()
