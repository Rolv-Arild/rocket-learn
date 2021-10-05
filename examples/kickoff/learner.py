# Preliminary setup for serious crowd-sourced model
# Exact setup should probably be in different repo
import os
from typing import Any

import numpy as np
import torch
import torch.jit
import wandb
from earl_pytorch import EARLPerceiver, ControlsPredictorDiscrete
from redis import Redis
from torch import nn
from torch.nn import Linear, Sequential

from rocket_learn.utils.util import SplitLayer
from rlgym.envs import Match
from rlgym.utils import ObsBuilder, RewardFunction
from rlgym.utils.common_values import ORANGE_TEAM, BOOST_LOCATIONS, BLUE_TEAM, BALL_MAX_SPEED, CEILING_Z, CAR_MAX_SPEED, \
    BALL_RADIUS
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition, \
    TimeoutCondition
from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator, RedisRolloutWorker

tick_skip = 1


class ExpandAdvancedObs(AdvancedObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return np.expand_dims(obs, 0)


def KickoffTerminal():
    return [TimeoutCondition(round(5 * 120 / tick_skip)), GoalScoredCondition()]


class KickoffReward(RewardFunction):
    def __init__(self, win_w=10, accel_w=0.5, touch_w=1):
        self.win_w = win_w
        self.accel_w = accel_w
        self.touch_w = touch_w
        self.last_state = None
        self.current_state = None
        self.rewards = None
        self.blue_rewards = None
        self.orange_rewards = None

    def reset(self, initial_state: GameState):
        self.last_state = None
        self.current_state = initial_state
        self.rewards = 0

    def _maybe_update_rewards(self, state: GameState, final=False):
        if state == self.current_state:
            return
        self.last_state = self.current_state
        self.current_state = state
        self.rewards = 0

        for old_p, new_p in zip(self.last_state.players, self.current_state.players):
            assert old_p.car_id == new_p.car_id
            diff_vel = (new_p.car_data.linear_velocity - old_p.car_data.linear_velocity) / CAR_MAX_SPEED
            diff_pos = self.current_state.ball.position - new_p.car_data.position
            rew = self.accel_w * np.dot(diff_vel, diff_pos / np.linalg.norm(diff_pos))

            if new_p.ball_touched and (self.last_state.ball.position[:2] == 0).all():  # Only first touch
                rew += self.touch_w

            if new_p.team_num == BLUE_TEAM:
                self.rewards += rew
            else:
                self.rewards -= rew

        if final and abs(state.ball.position[1]) > BALL_RADIUS:
            self.rewards += self.win_w * (2 * (state.ball.position[1] > 0) - 1)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self._maybe_update_rewards(state, final=True)
        return self.rewards if player.team_num == BLUE_TEAM else -self.rewards

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self._maybe_update_rewards(state, final=False)
        return self.rewards if player.team_num == BLUE_TEAM else -self.rewards


def get_match():
    return Match(
        reward_function=KickoffReward(),
        terminal_conditions=KickoffTerminal(),
        obs_builder=ExpandAdvancedObs(),
        state_setter=DefaultState(),
        self_play=True,
        team_size=1,
        tick_skip=tick_skip
    )


def make_worker(host, name, limit_threads=True):
    if limit_threads:
        torch.set_num_threads(1)
    r = Redis(host=host, password="rocket-learn")
    return RedisRolloutWorker(r, name, get_match(), current_version_prob=.9).run()


if __name__ == "__main__":
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(project="rocket-learn", entity="rolv-arild")

    redis = Redis(password="rocket-learn")
    rollout_gen = RedisRolloutGenerator(redis, save_every=10, logger=logger)

    # jit models can't be pickled
    # ex_inp = (
    #     (torch.zeros((10, 1, 32)), torch.zeros((10, 1 + 6 + 34, 24)), torch.zeros((10, 1 + 6 + 34))),)  # q, kv, mask
    # critic = torch.jit.trace(
    #     func=Necto(EARLPerceiver(128, query_features=32, key_value_features=24), Linear(128, 1)),
    #     example_inputs=ex_inp
    # )
    # actor = torch.jit.trace(
    #     func=Necto(EARLPerceiver(256, query_features=32, key_value_features=24), ControlsPredictorDiscrete(256)),
    #     example_inputs=ex_inp
    # )
    critic = Sequential(Linear(107, 128), Linear(128, 64), Linear(64, 32), Linear(32, 1))
    actor = DiscretePolicy(Sequential(Linear(107, 128), Linear(128, 64), Linear(64, 32), Linear(32, 21), SplitLayer()))

    lr = 5e-5
    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": lr},
        {"params": critic.parameters(), "lr": lr}
    ])

    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)

    alg = PPO(
        rollout_gen,
        agent,
        ent_coef=0.01,
        n_steps=1_00_000,
        batch_size=20_000,
        minibatch_size=10_000,
        epochs=10,
        gamma=599 / 600,  # 5 second horizon
        logger=logger,
    )

    log_dir = "E:\\log_directory\\"
    repo_dir = "E:\\repo_directory\\"

    alg.run(epochs_per_save=10, save_dir="ppos")
