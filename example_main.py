import os

import torch
import wandb
from torch import nn

from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions.common_rewards import VelocityReward
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rocket_learn.algorithms.ppo import PPO
from rocket_learn.agents.ppo_agent import PPOAgent
from rocket_learn.rollout_generators.redis_rolloutgenerator import RedisRolloutGenerator


class SplitLayer(nn.Module):
    def __init__(self, splits=None):
        super().__init__()
        if splits is not None:
            self.splits = splits
        else:
            self.splits = (3, 3, 3, 3, 3, 2, 2, 2)

    def forward(self, x):
        return torch.split(x, self.splits, dim=-1)


def get_match_args():
    return dict(
        game_speed=100,
        self_play=True,
        team_size=3,
        obs_builder=AdvancedObs(),
        terminal_conditions=[TimeoutCondition(10 * 120 // 8)],
        reward_fn=VelocityReward(negative=True),
    )


self_play = True

state_dim = 231
print(state_dim)
action_dim = 8

shared = nn.Sequential(
    nn.Linear(state_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
)

critic = nn.Sequential(
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)

actor = nn.Sequential(
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 21),
    SplitLayer()
)

if __name__ == '__main__':
    # rollout_gen = SimpleRolloutGenerator(None, **get_match_args())
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(project="rocket-learn", entity="rolv-arild")

    rollout_gen = RedisRolloutGenerator(password="rocket-learn", logger=logger, save_every=1)

    agent = PPOAgent(actor, critic, shared)
    alg = PPO(rollout_gen, agent, n_steps=5000, batch_size=500, lr_critic=3e-4, lr_actor=3e-4, epochs=10, logger=logger)
    # rollout_gen.agent = alg.agent
    # rl_path = "C:\\EpicGames\\rocketleague\\Binaries\\Win64\\RocketLeague.exe"
    log_dir = "E:\\log_directory\\"
    repo_dir = "E:\\repo_directory\\"

    alg.run()
