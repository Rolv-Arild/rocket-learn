from typing import Optional, Tuple, Dict, Union, List

import gymnasium
import numpy as np
import rlgym
from pettingzoo.utils.env import ObsDict, ActionDict
from rlgym.gym import Gym
from rlgym.utils.common_values import BLUE_TEAM
from rlgym.utils.gamestates import GameState

from rocket_learn.envs.rocket_league import RocketLeague
from rocket_learn.utils.truncation import TerminalTruncatedCondition
from rocket_learn.utils.gamestate_encoding import encode_gamestate


class RLGym(RocketLeague):
    def __init__(self, env: Gym):
        super().__init__(blue=env._match._team_size, orange=env._match._team_size)  # noqa
        self._env = env
        self._state = None

    @classmethod
    def make(cls, *args, **kwargs):
        return cls(rlgym.make(*args, **kwargs))

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> ObsDict:
        self._env.update_settings(game_speed=options.get("game_speed", None),
                                  gravity=options.get("gravity", None),
                                  boost_consumption=options.get("boost_consumption", None))
        obs, info = self._env.reset(return_info=True)
        state: GameState = info["state"]

        b = o = 0
        player_ids = []
        for player in state.players:
            if player.team_num == BLUE_TEAM:
                player_ids.append(f"blue-{b}")
                b += 1
            else:
                player_ids.append(f"orange-{o}")
                o += 1
        self.agents = player_ids

        obs = dict(zip(self.agents, obs))

        self._state = state

        return obs, info if return_info else obs  # noqa ObsDict does not account for info

    def seed(self, seed=None):
        pass

    def is_truncated(self, state):
        for condition in self._env._match._terminal_conditions:  # noqa
            if isinstance(condition, TerminalTruncatedCondition) \
                    and condition.is_truncated(state):
                return True
        return False

    def step(self, actions: ActionDict) -> Tuple[
        ObsDict, Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]
    ]:
        obs, reward, done, info = self._env.step([actions[agent] for agent in self.agents])

        state = info["state"]
        truncated = self.is_truncated(info["state"])

        self._state = state

        obs = dict(zip(self.agents, obs))
        reward = dict(zip(self.agents, reward))
        done = {a: done for a in self.agents}
        truncated = {a: truncated for a in self.agents}
        info = {a: info for a in self.agents}

        return obs, reward, done, truncated, info

    def state(self) -> np.ndarray:
        return encode_gamestate(self._state)

    def render(self) -> Union[None, np.ndarray, str, List]:
        return None

    def observation_space(self, agent: str) -> gymnasium.spaces.Space:
        space = self._env._match._obs_builder.get_obs_space()  # noqa
        if isinstance(space, dict):
            return space[agent]
        return space

    def action_space(self, agent: str) -> gymnasium.spaces.Space:
        space = self._env._match._action_parser.get_action_space()  # noqa
        if isinstance(space, dict):
            return space[agent]
        return space

    @property
    def game_speed(self):
        return self._env._match._gamespeed  # noqa

    @game_speed.setter
    def game_speed(self, value):
        self._env.update_settings(game_speed=value)

    @property
    def gravity(self):
        return self._env._match._gravity  # noqa

    @gravity.setter
    def gravity(self, value):
        self._env.update_settings(gravity=value)

    @property
    def boost_consumption(self):
        return self._env._match._boost_consumption  # noqa

    @boost_consumption.setter
    def boost_consumption(self, value):
        self._env.update_settings(boost_consumption=value)
