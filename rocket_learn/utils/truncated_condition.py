from abc import ABC

from rlgym_sim.utils import TerminalCondition
from rlgym_sim.utils.gamestates import GameState


class TruncatedCondition(TerminalCondition, ABC):
    def is_truncated(self, current_state: GameState):
        raise NotImplementedError


class TerminalToTruncatedWrapper(TruncatedCondition):
    def __init__(self, condition: TerminalCondition):
        super().__init__()
        self.condition = condition

    def is_truncated(self, current_state: GameState):
        return self.condition.is_terminal(current_state)

    def reset(self, initial_state: GameState):
        self.condition.reset(initial_state)

    def is_terminal(self, current_state: GameState) -> bool:
        return False
