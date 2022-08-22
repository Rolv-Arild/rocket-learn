import os

import numpy as np

import torch
import torch.nn.functional as F

from rocket_learn.agent.pretrained_policy import HardcodedAgent
from pretrained_agents.necto.necto_v1_obs import NectoV1Obs

from rlgym.utils.gamestates import GameState

import copy


class NectoV1(HardcodedAgent):
    def __init__(self, model_string, n_players):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.actor = torch.jit.load(os.path.join(cur_dir, model_string))
        self.obs_builder = NectoV1Obs(n_players=n_players)
        self.previous_action = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    def act(self, state: GameState, player_index: int):
        player = state.players[player_index]
        teammates = [p for p in state.players if p.team_num == player.team_num and p != player]
        opponents = [p for p in state.players if p.team_num != player.team_num]

        necto_gamestate: GameState = copy.deepcopy(state)
        necto_gamestate.players = [player] + teammates + opponents

        self.obs_builder.reset(necto_gamestate)
        obs = self.obs_builder.build_obs(player, necto_gamestate, self.previous_action)

        obs = tuple(torch.from_numpy(s).float() for s in obs)
        with torch.no_grad():
            out, _ = self.actor(obs)

        max_shape = max(o.shape[-1] for o in out)
        logits = torch.stack(
            [
                l
                if l.shape[-1] == max_shape
                else F.pad(l, pad=(0, max_shape - l.shape[-1]), value=float("-inf"))
                for l in out
            ]
        ).swapdims(0, 1).squeeze()

        actions = np.argmax(logits, axis=-1)

        actions = actions.reshape((-1, 5))
        actions[:, 0] = actions[:, 0] - 1
        actions[:, 1] = actions[:, 1] - 1

        parsed = np.zeros((actions.shape[0], 8))
        parsed[:, 0] = actions[:, 0]  # throttle
        parsed[:, 1] = actions[:, 1]  # steer
        parsed[:, 2] = actions[:, 0]  # pitch
        parsed[:, 3] = actions[:, 1] * (1 - actions[:, 4])  # yaw
        parsed[:, 4] = actions[:, 1] * actions[:, 4]  # roll
        parsed[:, 5] = actions[:, 2]  # jump
        parsed[:, 6] = actions[:, 3]  # boost
        parsed[:, 7] = actions[:, 4]  # handbrake

        self.previous_action = parsed[0]
        return parsed[0]
