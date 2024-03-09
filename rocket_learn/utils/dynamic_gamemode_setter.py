from rlgym_sim.utils import StateSetter
from rlgym_sim.utils.state_setters import StateWrapper


class DynamicGMSetter(StateSetter):
    def __init__(self, setter: StateSetter):
        self.setter = setter
        self.blue = 0
        self.orange = 0

    def set_team_size(self, blue=None, orange=None):
        if blue is not None:
            self.blue = blue
        if orange is not None:
            self.orange = orange

    def build_wrapper(self, max_team_size: int, spawn_opponents: bool) -> StateWrapper:
        assert self.blue <= max_team_size and self.orange <= max_team_size
        return StateWrapper(self.blue, self.orange)

    def reset(self, state_wrapper: StateWrapper):
        self.setter.reset(state_wrapper)
