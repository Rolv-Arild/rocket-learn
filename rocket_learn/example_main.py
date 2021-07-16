import torch
import torch.nn as nn
from learner import PPO
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym.utils.reward_functions.common_rewards import VelocityReward
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
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
        self_play=True,
        team_size=1,
        obs_builder=AdvancedObs(),
        terminal_conditions=[TimeoutCondition(600)],
        reward_function=VelocityReward(negative=True)
    )


state_dim = 30
action_dim = 8

# example pytorch stuff, delete later
actor = nn.Sequential(
    nn.Linear(state_dim, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 21),
    SplitLayer()
)

# critic
critic = nn.Sequential(
    nn.Linear(state_dim, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
)

# make sure to add this to avoid a torch multiprocessing bug
# https://github.com/pytorch/pytorch/issues/5858
if __name__ == '__main__':
    rollout_gen = SimpleRolloutGenerator(None, team_size=1, self_play=True)
    alg = PPO(rollout_gen, actor, critic)
    rollout_gen.agent = alg.agent
    # rl_path = "C:\\EpicGames\\rocketleague\\Binaries\\Win64\\RocketLeague.exe"
    log_dir = "C:\\log_directory\\"
    repo_dir = "C:\\repo_directory\\"

    alg.run()
