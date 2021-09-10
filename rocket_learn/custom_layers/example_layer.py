import torch
from torch import nn


class SplitLayer(nn.Module):
    def __init__(self, splits=None):
        super().__init__()
        if splits is not None:
            self.splits = splits
        else:
            self.splits = (3, 3, 3, 3, 3, 2, 2, 2)

    def forward(self, x):
        return torch.split(x, self.splits, dim=-1)


class MlpLayer:

    def __init__(
            self, num_layers_critic: int = 2,
            num_layers_actor: int = 2,
            feature_size_critic: int = 256,
            feature_size_actor: int = 256,
            include_shared: bool = False,
            num_layers_shared: int = 0,
            feature_size_shared: int = 0
    ):
        self.num_layers_critic = num_layers_critic
        self.feature_size_critic = feature_size_critic
        self.feature_size_actor = feature_size_actor
        self.num_layers_actor = num_layers_actor
        if include_shared:
            self.num_layers_shared = num_layers_shared
            self.feature_size_shared = feature_size_shared
            assert num_layers_shared > 0, "num_layers_shared needs to be greater than 0 when include_shared is True"

    def get_shared_layers(self):
        pass
    def get_critic_layers(self):
        pass
    def get_actor_layers(self):
        pass


    def _build_sequential_layer(self):
        pass

    # shared = nn.Sequential(
    #     nn.Linear(state_dim, 256),
    #     nn.ReLU(),
    #     nn.Linear(256, 128),
    #     nn.ReLU(),
    # )
    #
    # critic = nn.Sequential(
    #     nn.Linear(128, 128),
    #     nn.ReLU(),
    #     nn.Linear(128, 1)
    # )
    #
    # actor = nn.Sequential(
    #     nn.Linear(128, 128),
    #     nn.ReLU(),
    #     nn.Linear(128, 21),
    #     SplitLayer()
    # )