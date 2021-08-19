import torch
from torch import nn
import torch.nn.functional as F
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions.common_rewards import TouchBallReward, VelocityReward
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rocket_learn.learner import PPO
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.rollout_generator.simple_rollout_generator import SimpleRolloutGenerator


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

critic = nn.Sequential(
    nn.Linear(state_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 1)
)

actor = nn.Sequential(
    nn.Linear(state_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 21),
    SplitLayer()
)

if __name__ == '__main__':
    # rollout_gen = SimpleRolloutGenerator(None, **get_match_args())
    rollout_gen = RedisRolloutGenerator()

    alg = PPO(rollout_gen, actor, critic, n_steps=5000, batch_size=500, lr_critic=3e-4, lr_actor=3e-4, epochs=10)
    # rollout_gen.agent = alg.agent
    # rl_path = "C:\\EpicGames\\rocketleague\\Binaries\\Win64\\RocketLeague.exe"
    log_dir = "E:\\log_directory\\"
    repo_dir = "E:\\repo_directory\\"

    alg.run()
