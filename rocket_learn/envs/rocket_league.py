from typing import Optional, Tuple, Dict, Union, List

import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import ObsDict, ActionDict


class RocketLeague(ParallelEnv):
    def __init__(self, max_blue, max_orange):
        self.possible_agents = [f"{side}-{n}"
                                for side in ("blue", "orange")
                                for n in range(max_blue if side == "blue" else max_orange)]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> ObsDict:
        raise NotImplementedError

    def seed(self, seed=None):
        raise NotImplementedError

    def step(self, actions: ActionDict) -> Tuple[
        ObsDict, Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]
    ]:
        raise NotImplementedError

    def render(self) -> Union[None, np.ndarray, str, List]:
        raise NotImplementedError

    def state(self) -> np.ndarray:
        raise NotImplementedError
