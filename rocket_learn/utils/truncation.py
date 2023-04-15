from abc import ABC

from rlgym.utils import TerminalCondition
from rlgym.utils.gamestates import GameState


class TerminalTruncatedCondition(TerminalCondition, ABC):
    def is_truncated(self, current_state: GameState) -> bool:
        raise NotImplementedError


class TruncateWrapper(TerminalTruncatedCondition):
    # Returns result of is_terminal in is_truncated, and False in is_terminal
    def __init__(self, condition: TerminalCondition):
        super().__init__()
        self.condition = condition

    def reset(self, initial_state: GameState):
        self.condition.reset(initial_state)

    def is_terminal(self, current_state: GameState) -> bool:
        return False

    def is_truncated(self, current_state: GameState) -> bool:
        return self.condition.is_terminal(current_state)
