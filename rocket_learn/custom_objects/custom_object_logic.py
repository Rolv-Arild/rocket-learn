from typing import Dict, Any

from rlgym.rocket_league.api import GameState


class CustomObjectLogic:
    def reset(self, initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        raise NotImplementedError

    def step(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        raise NotImplementedError
