import torch
from torch import nn
import torch.nn.functional as F

from rocket_learn.learner import PPO
from rocket_learn.rollout_generator.simple_rollout_generator import SimpleRolloutGenerator


class SplitLayer(nn.Module):
    def __init__(self, splits=None):
        super().__init__()
        if splits is not None:
            self.splits = splits
        else:
            self.splits = (3, 3, 3, 3)

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
        reward_function=VelocityReward(negative=True)
    )


self_play = True

state_dim = 8
print(state_dim)
action_dim = 4

critic = nn.Sequential(
    nn.Linear(state_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 1)
)

actor = nn.Sequential(
    nn.Linear(state_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, action_dim),
    SplitLayer((4,))
)

if __name__ == '__main__':
    rollout_gen = SimpleRolloutGenerator(None, team_size=1, self_play=True)

    alg = PPO(rollout_gen, actor, critic, batch_size=32, n_steps=4096, lr_critic=.001, lr_actor=.001, epochs=1)
    rollout_gen.agent = alg.agent
    # rl_path = "C:\\EpicGames\\rocketleague\\Binaries\\Win64\\RocketLeague.exe"
    log_dir = "E:\\log_directory\\"
    repo_dir = "E:\\repo_directory\\"

    alg.run()
