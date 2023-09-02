from typing import List

from rlgym.api import AgentID
from rlgym.rocket_league.api import GameState

from rocket_learn.custom_objects.scoreboard.util import TICKS_PER_SECOND
from rocket_learn.stat_trackers.stat_tracker import StatTracker


class TimeoutRate(StatTracker):
    def __init__(self):
        super().__init__("Timeout rate", "mean")

    def collect(self, agent_ids: List[AgentID], game_states: List[GameState]):
        timed_out = not game_states[-1].goal_scored

        return int(timed_out)


class EpisodeLength(StatTracker):
    def __init__(self):
        super().__init__("Episode length (s)", "mean")

    def collect(self, agent_ids: List[AgentID], game_states: List[GameState]):
        ticks = game_states[-1].tick_count - game_states[0].tick_count

        return ticks * len(agent_ids) / TICKS_PER_SECOND
