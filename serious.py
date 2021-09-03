# Preliminary setup for serious crowd-sourced model
# Exact setup should probably be in different repo
import os
import random
from typing import Any

import numpy as np
import torch.jit
from earl_pytorch import EARLPerceiver, ControlsPredictorDiscrete
from rlgym_tools.extra_rewards.distribute_rewards import DistributeRewards
from torch.nn import Sequential, Linear

import wandb
from rlgym.envs import Match
from rlgym.utils import ObsBuilder, TerminalCondition, RewardFunction, StateSetter
from rlgym.utils.common_values import ORANGE_TEAM, BOOST_LOCATIONS, BLUE_TEAM, BLUE_GOAL_BACK, ORANGE_GOAL_BACK, \
    BALL_MAX_SPEED
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.math import cosine_similarity, scalar_projection
from rlgym.utils.reward_functions.common_rewards import EventReward
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
from rocket_learn.ppo import PPOAgent, PPO
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator


class SeriousObsBuilder(ObsBuilder):
    _boost_locations = np.array(BOOST_LOCATIONS)
    _invert = np.array([1] * 5 + [-1, -1, 1] * 5 + [1] * 4)
    _norm = np.array([1.] * 5 + [2300] * 6 + [1] * 6 + [5.5] * 3 + [1] * 4)

    def __init__(self, n_players=6, tick_skip=8):
        super().__init__()
        self.n_players = n_players
        self.demo_timers = None
        self.boost_timers = None
        self.current_state = None
        self.current_qkv = None
        self.current_mask = None
        self.tick_skip = tick_skip

    def reset(self, initial_state: GameState):
        self.demo_timers = np.zeros(len(initial_state.players))
        self.boost_timers = np.zeros(len(initial_state.boost_pads))

    def update_if_new_state(self, state: GameState):
        if self.current_state == state:  # No need to update
            return

        qkv = np.zeros((1 + self.n_players + len(state.boost_pads), 24))  # Ball, players, boosts

        # Add ball
        n = 0
        ball = state.ball
        qkv[0, 3] = 1  # is_ball
        qkv[0, 5:8] = ball.position
        qkv[0, 8:11] = ball.linear_velocity
        qkv[0, 17:20] = ball.angular_velocity

        # Add players
        n += 1
        demos = np.zeros(len(state.players))  # Which players are currently demoed
        for player in state.players:
            if player.team_num == BLUE_TEAM:
                qkv[n, 1] = 1  # is_teammate
            else:
                qkv[n, 2] = 1  # is_opponent
            car_data = player.car_data
            qkv[n, 5:8] = car_data.position
            qkv[n, 8:11] = car_data.linear_velocity
            qkv[n, 11:14] = car_data.forward()
            qkv[n, 14:17] = car_data.up()
            qkv[n, 17:20] = car_data.angular_velocity
            qkv[n, 20] = player.boost_amount
            #             qkv[n, 21] = player.is_demoed
            demos[n - 1] = player.is_demoed  # Keep track for demo timer
            qkv[n, 22] = player.on_ground
            qkv[n, 23] = player.has_flip
            n += 1

        # Add boost pads
        n = 1 + self.n_players
        boost_pads = state.boost_pads
        qkv[n:, 4] = 1  # is_boost
        qkv[n:, 5:8] = self._boost_locations
        qkv[n:, 20] = 0.12 + 0.88 * (self._boost_locations[:, 2] > 72)  # Boost amount
        #         qkv[n:, 21] = boost_pads

        # Boost and demo timers
        new_boost_grabs = boost_pads & (self.boost_timers == 0)  # New boost grabs since last frame
        self.boost_timers[new_boost_grabs] = 0.4 + 0.6 * (self._boost_locations[new_boost_grabs, 2] > 72)
        self.boost_timers *= boost_pads  # Make sure we have zeros right
        qkv[1 + self.n_players:, 21] = self.boost_timers
        self.boost_timers -= self.tick_skip / 1200  # Pre-normalized, 120 fps for 10 seconds
        self.boost_timers[self.boost_timers < 0] = 0

        new_demos = demos & (self.demo_timers == 0)
        self.demo_timers[new_demos] = 0.3
        self.demo_timers *= demos
        qkv[1: 1 + self.n_players, 21] = self.demo_timers
        self.demo_timers -= self.tick_skip / 1200
        self.demo_timers[self.demo_timers < 0] = 0

        # Store results
        self.current_qkv = qkv / self._norm
        mask = np.zeros(qkv.shape[0])
        mask[1 + len(state.players):1 + self.n_players] = 1
        self.current_mask = mask

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        self.update_if_new_state(state)
        invert = player.team_num == ORANGE_TEAM

        qkv = self.current_qkv.copy()
        mask = self.current_mask.copy()

        main_n = state.players.index(player)
        qkv[main_n, 0] = 1  # is_main
        if invert:
            qkv[:, (1, 2)] = qkv[:, (2, 1)]  # Swap blue/orange
            qkv *= self._invert  # Negate x and y values

        q = qkv[main_n, :]
        q = np.expand_dims(np.concatenate((q, previous_action), axis=0), axis=0)
        # kv = np.delete(qkv, main_n, axis=0)  # Delete main? Watch masking
        kv = qkv

        # With EARLPerceiver we can use relative coords+vel(+more?) for key/value tensor, might be smart
        kv[:, 5:11] -= q[:, 5:11]
        return q, kv, mask


def SeriousTerminalCondition(tick_skip=8):
    return [NoTouchTimeoutCondition(round(30 * tick_skip / 120)), GoalScoredCondition()]


class SeriousRewardFunction(RewardFunction):
    def __init__(self):
        self.last_state = None
        self.current_state = None

    # Something like DistributeRewards(EventReward(goal=4, shot=4, save=4, demo=4, touch=1))
    # but find a way to reduce dribble abuse
    # Also add std/max/min rewards to log so we can actually see progress
    def reset(self, initial_state: GameState):
        self.last_state = None
        self.current_state = initial_state

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state != self.current_state:
            self.last_state = self.current_state
            self.current_state = state
        if self.last_state is None:
            return 0

        rew = 0
        for p in self.last_state.players:
            if p.car_id == player.car_id:
                rew += 10 * (player.match_goals - p.match_shots) + \
                       5 * (player.match_shots - p.match_shots) + \
                       5 * (player.match_saves - p.match_saves) + \
                       5 * (player.match_demolishes - p.match_demolishes)
                break
        if player.ball_touched:
            target = np.array(BLUE_GOAL_BACK if player.team_num == BLUE_TEAM else ORANGE_GOAL_BACK)
            curr_vel = self.current_state.ball.linear_velocity
            last_vel = self.last_state.ball.linear_velocity
            rew += scalar_projection(curr_vel - last_vel, target - state.ball.position) / BALL_MAX_SPEED


class SeriousStateSetter(StateSetter):
    # Use anything other than DefaultState?
    # Random is useful at start since it has to actually learn where ball is (somewhat less necessary with relative obs)
    def reset(self, state_wrapper: StateWrapper):
        pass


def get_match():
    weights = (6, 3, 2)  # equal number of agents
    return Match(
        reward_function=SeriousRewardFunction(),
        terminal_conditions=SeriousTerminalCondition(),
        obs_builder=SeriousObsBuilder(),
        state_setter=SeriousStateSetter(),
        self_play=True,
        team_size=random.choices((1, 2, 3), weights)[0],  # Use mix of 1s, 2s and 3s?
    )


if __name__ == "__main__":
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(project="rocket-learn", entity="rolv-arild")

    rollout_gen = RedisRolloutGenerator(password="rocket-learn", logger=logger, save_every=1)

    d = 256
    actor = torch.jit.trace(Sequential(Linear(d, d), ControlsPredictorDiscrete(d)), torch.zeros(1, 1, d))
    critic = torch.jit.trace(Sequential(Linear(d, d), Linear(d, 1)), torch.zeros(1, 1, d))
    shared = torch.jit.trace(EARLPerceiver(d, query_features=32, key_value_features=24),
                             (torch.zeros(10, 1, 32), torch.zeros(10, 1+6+34, 24), torch.zeros(10, 1+6+34)))

    agent = PPOAgent(actor=actor, critic=critic, shared=shared)

    lr = 1e-5
    alg = PPO(
        rollout_gen,
        agent,
        n_steps=1_000_000,
        batch_size=10_000,
        lr_critic=lr,
        lr_actor=lr,
        lr_shared=lr,
        epochs=10,
        logger=logger
    )

    log_dir = "E:\\log_directory\\"
    repo_dir = "E:\\repo_directory\\"

    alg.run()
