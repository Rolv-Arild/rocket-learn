import numpy as np

from rocket_learn.utils.gamestate_encoding import StateConstants
from rocket_learn.utils.scoreboard import TICKS_PER_SECOND
from rocket_learn.utils.stat_trackers.stat_tracker import StatTracker


class DemosPerMinute(StatTracker):
    def __init__(self, tick_skip):
        super().__init__("demos_per_minute")
        self.count = 0
        self.total_demos = 0
        self.tick_skip = tick_skip

    def reset(self):
        self.count = 0
        self.total_demos = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        players = gamestates[:, StateConstants.PLAYERS]
        demos = players[:, StateConstants.MATCH_DEMOLISHES]
        self.count += demos.size
        demos = np.clip(demos[-1] - demos[0],
                        0, None)
        self.total_demos += np.sum(demos)

    def get_stat(self):
        ticks = self.count
        seconds = self.tick_skip * ticks / TICKS_PER_SECOND
        minutes = seconds / 60
        return self.total_demos / (minutes or 1)


class TouchesPerMinute(StatTracker):
    def __init__(self, tick_skip):
        super().__init__("touches_per_minute")
        self.count = 0
        self.total_touches = 0
        self.tick_skip = tick_skip

    def reset(self):
        self.count = 0
        self.total_touches = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        players = gamestates[:, StateConstants.PLAYERS]
        is_touch = players[:, StateConstants.BALL_TOUCHED]

        self.total_touches += np.sum(is_touch)
        self.count += is_touch.size

    def get_stat(self):
        ticks = self.count
        seconds = self.tick_skip * ticks / TICKS_PER_SECOND
        minutes = seconds / 60
        return self.total_touches / (minutes or 1)


class SavesPerMinute(StatTracker):
    def __init__(self, tick_skip):
        super().__init__("saves_per_minute")
        self.count = 0
        self.total_saves = 0
        self.tick_skip = tick_skip

    def reset(self):
        self.count = 0
        self.total_saves = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        players = gamestates[:, StateConstants.PLAYERS]

        saves = players[:, StateConstants.MATCH_SAVES]
        self.count += saves.size
        saves = np.clip(saves[-1] - saves[0],
                        0, None)
        self.total_saves += np.sum(saves)

    def get_stat(self):
        ticks = self.count
        seconds = self.tick_skip * ticks / TICKS_PER_SECOND
        minutes = seconds / 60
        return self.total_saves / (minutes or 1)


class ShotsPerMinute(StatTracker):
    def __init__(self, tick_skip):
        super().__init__("shots_per_minute")
        self.count = 0
        self.total_shots = 0
        self.tick_skip = tick_skip

    def reset(self):
        self.count = 0
        self.total_shots = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        players = gamestates[:, StateConstants.PLAYERS]

        shots = players[:, StateConstants.MATCH_SHOTS]
        self.count += shots.size
        shots = np.clip(shots[-1] - shots[0],
                        0, None)
        self.total_shots += np.sum(shots)

    def get_stat(self):
        ticks = self.count
        seconds = self.tick_skip * ticks / TICKS_PER_SECOND
        minutes = seconds / 60
        return self.total_shots / (minutes or 1)


class BoostUsedPerMinute(StatTracker):
    def __init__(self, tick_skip):
        super().__init__("boost_used_per_minute")
        self.count = 0
        self.usage = 0
        self.tick_skip = tick_skip

    def reset(self):
        self.count = 0
        self.usage = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        players = gamestates[:, StateConstants.PLAYERS]  # Shape: (n_ticks, n_players * n_features)
        boost = players[:, StateConstants.BOOST_AMOUNT]  # Shape: (n_ticks, n_players)
        is_limited = (0 <= boost) & (boost <= 1)
        boost_diff = np.diff(boost[is_limited], axis=0)  # Shape: (n_ticks - 1, n_players)
        self.usage -= np.sum(boost_diff[boost_diff < 0])  # Subtract because we want to add the absolute value
        self.count += boost_diff.size

    def get_stat(self):
        ticks = self.count
        seconds = self.tick_skip * ticks / TICKS_PER_SECOND
        minutes = seconds / 60
        return self.usage / (minutes or 1)
