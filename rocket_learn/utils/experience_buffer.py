from dataclasses import dataclass

import numpy as np

from rocket_learn.utils.util import calculate_advantages, conditional_stack


@dataclass(slots=True, init=False)
class ExperienceBuffer:
    """
    A buffer to store trajectories for PPO.
    """

    _observations = []
    _actions = []
    _rewards = []
    _values = []
    _log_probs = []
    _terminations = []
    _truncations = []

    # Stacked versions of the above, so we don't have to stack them every time they are accessed
    _stacked_observations = None
    _stacked_actions = None
    _stacked_rewards = None
    _stacked_values = None
    _stacked_log_probs = None
    _stacked_terminations = None
    _stacked_truncations = None

    def _clear_stacks(self):
        self._stacked_observations = None
        self._stacked_actions = None
        self._stacked_rewards = None
        self._stacked_values = None
        self._stacked_log_probs = None
        self._stacked_terminations = None
        self._stacked_truncations = None

    def clear(self):
        self._observations = []
        self._actions = []
        self._rewards = []
        self._values = []
        self._log_probs = []
        self._terminations = []
        self._truncations = []

        self._clear_stacks()

    def add_step(self, observation, action, reward, value, log_prob, is_terminal, is_truncated):
        self._observations.append(observation)
        self._actions.append(action)
        self._rewards.append(reward)
        self._values.append(value)
        self._log_probs.append(log_prob)
        self._terminations.append(is_terminal)
        self._truncations.append(is_truncated)

        self._clear_stacks()

    @property
    def observations(self):
        if self._stacked_observations is None:
            stacked = conditional_stack(self._observations)
            self._stacked_observations = stacked
        return self._stacked_observations

    @property
    def actions(self):
        if self._stacked_actions is None:
            stacked = conditional_stack(self._actions)
            self._stacked_actions = stacked
        return self._stacked_actions

    @property
    def rewards(self):
        if self._stacked_rewards is None:
            stacked = conditional_stack(self._rewards)
            self._stacked_rewards = stacked
        return self._stacked_rewards

    @property
    def values(self):
        if self._stacked_values is None:
            stacked = conditional_stack(self._values)
            self._stacked_values = stacked
        return self._stacked_values

    @property
    def log_probs(self):
        if self._stacked_log_probs is None:
            stacked = conditional_stack(self._log_probs)
            self._stacked_log_probs = stacked
        return self._stacked_log_probs

    @property
    def terminations(self):
        if self._stacked_terminations is None:
            stacked = conditional_stack(self._terminations)
            self._stacked_terminations = stacked
        return self._stacked_terminations

    @property
    def truncations(self):
        if self._stacked_truncations is None:
            stacked = conditional_stack(self._truncations)
            self._stacked_truncations = stacked
        return self._stacked_truncations

    def advantages(self, gamma, gae_lambda):
        return calculate_advantages(gamma, gae_lambda, self.rewards, self.values)

    def __len__(self):
        return len(self._observations)

    @classmethod
    def from_fixed_buffer(cls, fixed_buffer):
        obj = cls()
        obj._observations = list(fixed_buffer.observations)
        obj._actions = list(fixed_buffer.actions)
        obj._rewards = list(fixed_buffer.rewards)
        obj._values = list(fixed_buffer.values)
        obj._log_probs = list(fixed_buffer.log_probs)
        obj._terminations = list(fixed_buffer.terminations)
        obj._truncations = list(fixed_buffer.truncations)
        return obj

    def to_fixed_buffer(self):
        return self[:]

    def __getitem__(self, item):
        return FixedBuffer(
            observations=self.observations[item],
            actions=self.actions[item],
            rewards=self.rewards[item],
            values=self.values[item],
            log_probs=self.log_probs[item],
            terminations=self.terminations[item],
            truncations=self.truncations[item]
        )


@dataclass(slots=True, frozen=True)
class FixedBuffer:
    """
    A fixed-size buffer to store trajectories for PPO.
    """
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    log_probs: np.ndarray
    terminations: np.ndarray
    truncations: np.ndarray

    def __len__(self):
        return len(self.observations)

    def advantages(self, gamma, gae_lambda):
        return calculate_advantages(gamma, gae_lambda, self.rewards, self.values)

    def to_experience_buffer(self):
        return ExperienceBuffer.from_fixed_buffer(self)
