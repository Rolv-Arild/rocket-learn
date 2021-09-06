# Preliminary setup for serious crowd-sourced model
# Exact setup should probably be in different repo
import os
import random
from typing import Any

import numpy as np
import torch.jit
from earl_pytorch import EARLPerceiver, ControlsPredictorDiscrete
from redis import Redis
from torch import nn
from torch.nn import Sequential, Linear

import wandb
from rlgym.envs import Match
from rlgym.utils import ObsBuilder, RewardFunction
from rlgym.utils.common_values import ORANGE_TEAM, BOOST_LOCATIONS, BLUE_TEAM, BLUE_GOAL_BACK, ORANGE_GOAL_BACK, \
    BALL_MAX_SPEED, CEILING_Z
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.math import scalar_projection
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
from rocket_learn.ppo import PPOAgent, PPO
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator, RedisRolloutWorker

WORKER_COUNTER = "worker-counter"


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
        self.demo_timers = np.zeros(self.n_players)
        self.boost_timers = np.zeros(len(initial_state.boost_pads))
        # self.current_state = initial_state

    def _maybe_update_obs(self, state: GameState):
        if state == self.current_state:  # No need to update
            return

        if self.boost_timers is None:
            self.reset(state)
        else:
            self.current_state = state

        qkv = np.zeros((1, 1 + self.n_players + len(state.boost_pads), 24))  # Ball, players, boosts

        # Add ball
        n = 0
        ball = state.ball
        qkv[0, 0, 3] = 1  # is_ball
        qkv[0, 0, 5:8] = ball.position
        qkv[0, 0, 8:11] = ball.linear_velocity
        qkv[0, 0, 17:20] = ball.angular_velocity

        # Add players
        n += 1
        demos = np.zeros(self.n_players)  # Which players are currently demoed
        for player in state.players:
            if player.team_num == BLUE_TEAM:
                qkv[0, n, 1] = 1  # is_teammate
            else:
                qkv[0, n, 2] = 1  # is_opponent
            car_data = player.car_data
            qkv[0, n, 5:8] = car_data.position
            qkv[0, n, 8:11] = car_data.linear_velocity
            qkv[0, n, 11:14] = car_data.forward()
            qkv[0, n, 14:17] = car_data.up()
            qkv[0, n, 17:20] = car_data.angular_velocity
            qkv[0, n, 20] = player.boost_amount
            #             qkv[0, n, 21] = player.is_demoed
            demos[n - 1] = player.is_demoed  # Keep track for demo timer
            qkv[0, n, 22] = player.on_ground
            qkv[0, n, 23] = player.has_flip
            n += 1

        # Add boost pads
        n = 1 + self.n_players
        boost_pads = state.boost_pads
        qkv[0, n:, 4] = 1  # is_boost
        qkv[0, n:, 5:8] = self._boost_locations
        qkv[0, n:, 20] = 0.12 + 0.88 * (self._boost_locations[:, 2] > 72)  # Boost amount
        #         qkv[0, n:, 21] = boost_pads

        # Boost and demo timers
        new_boost_grabs = (boost_pads == 1) & (self.boost_timers == 0)  # New boost grabs since last frame
        self.boost_timers[new_boost_grabs] = 0.4 + 0.6 * (self._boost_locations[new_boost_grabs, 2] > 72)
        self.boost_timers *= boost_pads  # Make sure we have zeros right
        qkv[0, 1 + self.n_players:, 21] = self.boost_timers
        self.boost_timers -= self.tick_skip / 1200  # Pre-normalized, 120 fps for 10 seconds
        self.boost_timers[self.boost_timers < 0] = 0

        new_demos = (demos == 1) & (self.demo_timers == 0)
        self.demo_timers[new_demos] = 0.3
        self.demo_timers *= demos
        qkv[0, 1: 1 + self.n_players, 21] = self.demo_timers
        self.demo_timers -= self.tick_skip / 1200
        self.demo_timers[self.demo_timers < 0] = 0

        # Store results
        self.current_qkv = qkv / self._norm
        mask = np.zeros((1, qkv.shape[1]))
        mask[0, 1 + len(state.players):1 + self.n_players] = 1
        self.current_mask = mask

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        if self.boost_timers is None:
            return np.zeros(0)  # Obs space autodetect, make Aech happy
        self._maybe_update_obs(state)
        invert = player.team_num == ORANGE_TEAM

        qkv = self.current_qkv.copy()
        mask = self.current_mask.copy()

        main_n = state.players.index(player)
        qkv[0, main_n, 0] = 1  # is_main
        if invert:
            qkv[0, :, (1, 2)] = qkv[0, :, (2, 1)]  # Swap blue/orange
            qkv *= self._invert  # Negate x and y values

        q = qkv[0, main_n, :]
        q = np.expand_dims(np.concatenate((q, previous_action), axis=0), axis=(0, 1))
        # kv = np.delete(qkv, main_n, axis=0)  # Delete main? Watch masking
        kv = qkv

        # With EARLPerceiver we can use relative coords+vel(+more?) for key/value tensor, might be smart
        kv[0, :, 5:11] -= q[0, 0, 5:11]
        return q, kv, mask


def SeriousTerminalCondition(tick_skip=8):
    return [NoTouchTimeoutCondition(round(30 * 120 / tick_skip)), GoalScoredCondition()]


class SeriousRewardFunction(RewardFunction):
    def __init__(self, team_spirit=0.3, goal_w=10, shot_w=5, save_w=5, demo_w=5, boost_w=0.5):
        self.team_spirit = team_spirit
        self.last_state = None
        self.current_state = None
        self.rewards = None
        self.blue_rewards = None
        self.orange_rewards = None
        self.n = 0
        self.goal_w = goal_w
        self.shot_w = shot_w
        self.save_w = save_w
        self.demo_w = demo_w
        self.boost_w = boost_w

    def reset(self, initial_state: GameState):
        self.last_state = None
        self.current_state = initial_state
        self.rewards = np.zeros(len(initial_state.players))

    def _maybe_update_rewards(self, state: GameState):
        if state == self.current_state:
            return
        self.n = 0
        self.last_state = self.current_state
        self.current_state = state
        rewards = np.zeros(len(state.players))
        blue_mask = np.zeros_like(rewards, dtype=bool)
        orange_mask = np.zeros_like(rewards, dtype=bool)
        i = 0
        for old_p, new_p in zip(self.last_state.players, self.current_state.players):
            assert old_p.car_id == new_p.car_id
            rew = (self.goal_w * (new_p.match_goals - old_p.match_shots) +
                   self.shot_w * (new_p.match_shots - old_p.match_shots) +
                   self.save_w * (new_p.match_saves - old_p.match_saves) +
                   self.demo_w * (new_p.match_demolishes - old_p.match_demolishes) +
                   self.boost_w * max(new_p.boost_amount - old_p.boost_amount, 0))
            # Some napkin math: going around edge of field picking up 100 boost every second and gamma 0.995, skip 8
            # Discounted future reward in limit would be (0.5 / (1 * 15)) / (1 - 0.995) = 6.67 as a generous estimate
            # Pros are generally around maybe 400 bcpm, which would be 0.44 limit
            if new_p.ball_touched:
                target = np.array(BLUE_GOAL_BACK if new_p.team_num == BLUE_TEAM else ORANGE_GOAL_BACK)
                curr_vel = self.current_state.ball.linear_velocity
                last_vel = self.last_state.ball.linear_velocity
                # On ground it gets about 0.05 just for touching, as well as some extra for the speed it produces
                # Close to 20 in the limit with ball on top, but opponents should learn to challenge way before that
                rew += (state.ball.position[2] / CEILING_Z +
                        scalar_projection(curr_vel - last_vel, target - state.ball.position) / BALL_MAX_SPEED)

            rewards[i] = rew
            if new_p.team_num == BLUE_TEAM:
                blue_mask[i] = True
            else:
                orange_mask[i] = True
            i += 1

        blue_rewards = rewards[blue_mask]
        orange_rewards = rewards[orange_mask]
        blue_mean = np.nan_to_num(blue_rewards.mean())
        orange_mean = np.nan_to_num(orange_rewards.mean())
        self.rewards = np.zeros_like(rewards)
        self.rewards[blue_mask] = (1 - self.team_spirit) * blue_rewards + self.team_spirit * blue_mean - orange_mean
        self.rewards[orange_mask] = (1 - self.team_spirit) * orange_rewards + self.team_spirit * orange_mean - blue_mean

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self._maybe_update_rewards(state)
        rew = self.rewards[self.n]
        self.n += 1
        return rew


def SeriousStateSetter():
    # Use anything other than DefaultState?
    # Random is useful at start since it has to actually learn where ball is (somewhat less necessary with relative obs)
    return DefaultState()


def get_match(r):
    order = (1, 1, 2, 1, 1, 2, 3, 1, 1, 2, 3)  # Use mix of 1s, 2s and 3s?
    team_size = order[r % len(order)]
    return Match(
        reward_function=SeriousRewardFunction(),
        terminal_conditions=SeriousTerminalCondition(),
        obs_builder=SeriousObsBuilder(),
        state_setter=SeriousStateSetter(),
        self_play=True,
        team_size=team_size,
    )


class Necto(nn.Module):
    def __init__(self, earl, output):
        super().__init__()
        self.earl = earl
        self.output = output

    def forward(self, inp):
        q, kv, m = inp
        res = self.output(self.earl(q, kv, m))
        if isinstance(res, tuple):
            return tuple(torch.squeeze(r) for r in res)
        return torch.squeeze(res)


def make_worker(host, name):
    r = Redis(host=host, password="rocket-learn")
    w = r.incr(WORKER_COUNTER) - 1
    return RedisRolloutWorker(r, name, get_match(w)).run()


def collate(observations):
    transposed = tuple(zip(*observations))
    return tuple(torch.as_tensor(np.vstack(t)).float() for t in transposed)


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
    critic = Necto(EARLPerceiver(128, query_features=32, key_value_features=24), Linear(128, 1))
    actor = Necto(EARLPerceiver(256, query_features=32, key_value_features=24), ControlsPredictorDiscrete(256))

    agent = PPOAgent(actor=actor, critic=critic, collate_fn=collate)

    lr = 1e-5
    alg = PPO(
        rollout_gen,
        agent,
        n_steps=1_0_000,
        batch_size=10_000,
        lr_critic=lr,
        lr_actor=lr,
        # lr_shared=lr,
        epochs=10,
        logger=logger
    )

    log_dir = "E:\\log_directory\\"
    repo_dir = "E:\\repo_directory\\"

    alg.run()
