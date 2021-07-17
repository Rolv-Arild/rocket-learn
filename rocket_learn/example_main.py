import torch
from torch import nn
import torch.nn.functional as F
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions.common_rewards import TouchBallReward
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rocket_learn.learner import PPO
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
        random_resets=True,
        self_play=False,
        team_size=1,
        obs_builder=AdvancedObs(),
        terminal_conditions=[TimeoutCondition(10 * 120 // 8)],
        reward_function=TouchBallReward(1)
    )


self_play = True

state_dim = 30
print(state_dim)
action_dim = 8

critic = nn.Sequential(
    nn.Linear(state_dim, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
)

actor = nn.Sequential(
    nn.Linear(state_dim, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 21),
    SplitLayer()
)

if __name__ == '__main__':
    rollout_gen = SimpleRolloutGenerator(None, team_size=1, self_play=True)

    alg = PPO(rollout_gen, actor, critic, n_rollouts=36)
    rollout_gen.agent = alg.agent
    # rl_path = "C:\\EpicGames\\rocketleague\\Binaries\\Win64\\RocketLeague.exe"
    log_dir = "E:\\log_directory\\"
    repo_dir = "E:\\repo_directory\\"

    alg.run()
