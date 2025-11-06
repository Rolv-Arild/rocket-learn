import numpy as np

from rocket_learn.utils.gamestate_encoding import StateConstants
from rocket_learn.utils.stat_trackers.stat_tracker import StatTracker


class TimeoutRate(StatTracker):
    def __init__(self):
        super().__init__("timeout_rate")
        self.count = 0
        self.total_timeouts = 0

    def reset(self):
        self.count = 0
        self.total_timeouts = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        orange_diff = gamestates[-1, StateConstants.ORANGE_SCORE] - gamestates[0, StateConstants.ORANGE_SCORE]
        blue_diff = gamestates[-1, StateConstants.BLUE_SCORE] - gamestates[0, StateConstants.BLUE_SCORE]

        self.total_timeouts += ((orange_diff == 0) & (blue_diff == 0)).item()
        self.count += 1

    def get_stat(self):
        return self.total_timeouts / (self.count or 1)


class BehindBallRate(StatTracker):
    def __init__(self):
        super().__init__("behind_ball_rate")
        self.count = 0
        self.total_behind = 0

    def reset(self):
        self.count = 0
        self.total_behind = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        players = gamestates[:, StateConstants.PLAYERS]
        ball_y = gamestates[:, StateConstants.BALL_POSITION.start + 1]
        player_y = players[:, StateConstants.CAR_POS_Y]
        is_orange = players[:, StateConstants.TEAM_NUMS]
        behind = (2 * is_orange - 1) * (ball_y.reshape(-1, 1) - player_y) < 0

        self.total_behind += np.sum(behind)
        self.count += behind.size

    def get_stat(self):
        return self.total_behind / (self.count or 1)


class CarOnGroundRate(StatTracker):
    def __init__(self):
        super().__init__("on_ground_rate")
        self.count = 0
        self.total_ground = 0.0

    def reset(self):
        self.count = 0
        self.total_ground = 0.0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        on_ground = gamestates[:, StateConstants.ON_GROUND]

        self.total_ground += np.sum(on_ground)
        self.count += on_ground.size

    def get_stat(self):
        return 100 * self.total_ground / (self.count or 1)


class AirTouch(StatTracker):
    def __init__(self):
        super().__init__("air_touch_rate")
        self.count = 0
        self.total_touches = 0

    def reset(self):
        self.count = 0
        self.total_touches = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        players = gamestates[:, StateConstants.PLAYERS]
        is_touch = np.asarray([a * b for a, b in
                               zip(players[:, StateConstants.BALL_TOUCHED],
                                   np.invert(players[:, StateConstants.ON_GROUND].astype(bool)))])

        self.total_touches += np.sum(is_touch)
        self.count += is_touch.size

    def get_stat(self):
        return self.total_touches / (self.count or 1)
