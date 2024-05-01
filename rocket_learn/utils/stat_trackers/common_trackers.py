import numpy as np

from rocket_learn.utils.gamestate_encoding import StateConstants
from rocket_learn.utils.stat_trackers.stat_tracker import StatTracker


class AverageSpeed(StatTracker):
    def __init__(self):
        super().__init__("average_speed")
        self.count = 0
        self.total_speed = 0.0

    def reset(self):
        self.count = 0
        self.total_speed = 0.0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        players = gamestates[:, StateConstants.PLAYERS]
        xs = players[:, StateConstants.CAR_LINEAR_VEL_X]
        ys = players[:, StateConstants.CAR_LINEAR_VEL_Y]
        zs = players[:, StateConstants.CAR_LINEAR_VEL_Z]

        speeds = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
        self.total_speed += np.sum(speeds)
        self.count += speeds.size

    def get_stat(self):
        return self.total_speed / (self.count or 1)


class AverageEpisodeLength(StatTracker):
    def __init__(self):
        super().__init__("average_episode_length")
        self.count = 0
        self.total_length = 0

    def reset(self):
        self.count = 0
        self.total_length = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        self.total_length += gamestates.shape[0]
        self.count += 1

    def get_stat(self):
        return self.total_length / (self.count or 1)


class AverageBoost(StatTracker):
    def __init__(self):
        super().__init__("average_boost")
        self.count = 0
        self.total_boost = 0

    def reset(self):
        self.count = 0
        self.total_boost = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        players = gamestates[:, StateConstants.PLAYERS]
        boost = players[:, StateConstants.BOOST_AMOUNT]
        is_limited = (0 <= boost) & (boost <= 1)
        boost = boost[is_limited]
        self.total_boost += np.sum(boost)
        self.count += boost.size

    def get_stat(self):
        return self.total_boost / (self.count or 1)


class AverageTouchHeight(StatTracker):
    def __init__(self):
        super().__init__("average_touch_height")
        self.count = 0
        self.total_height = 0

    def reset(self):
        self.count = 0
        self.total_height = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        players = gamestates[:, StateConstants.PLAYERS]
        ball_z = gamestates[:, StateConstants.BALL_POSITION.start + 2]
        touch_heights = ball_z[players[:, StateConstants.BALL_TOUCHED].any(axis=1)]

        self.total_height += np.sum(touch_heights)
        self.count += touch_heights.size

    def get_stat(self):
        return self.total_height / (self.count or 1)


class AverageDistToBall(StatTracker):
    def __init__(self):
        super().__init__("average_distance_to_ball")
        self.count = 0
        self.total_dist = 0.0

    def reset(self):
        self.count = 0
        self.total_dist = 0.0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        players = gamestates[:, StateConstants.PLAYERS]
        ball = gamestates[:, StateConstants.BALL_POSITION]
        ball_x = ball[:, 0].reshape((-1, 1))
        ball_y = ball[:, 1].reshape((-1, 1))
        ball_z = ball[:, 2].reshape((-1, 1))
        xs = players[:, StateConstants.CAR_POS_X]
        ys = players[:, StateConstants.CAR_POS_Y]
        zs = players[:, StateConstants.CAR_POS_Z]

        dists = np.sqrt((ball_x - xs) ** 2 + (ball_y - ys) ** 2 + (ball_z - zs) ** 2)
        self.total_dist += np.sum(dists)
        self.count += dists.size

    def get_stat(self):
        return self.total_dist / (self.count or 1)


class AverageBallSpeed(StatTracker):
    def __init__(self):
        super().__init__("average_ball_speed")
        self.count = 0
        self.total_speed = 0.0

    def reset(self):
        self.count = 0
        self.total_speed = 0.0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        ball_speeds = gamestates[:, StateConstants.BALL_LINEAR_VELOCITY]
        xs = ball_speeds[:, 0]
        ys = ball_speeds[:, 1]
        zs = ball_speeds[:, 2]
        speeds = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
        self.total_speed += np.sum(speeds)
        self.count += speeds.size

    def get_stat(self):
        return self.total_speed / (self.count or 1)


class AverageBallHeight(StatTracker):
    def __init__(self):
        super().__init__("average_ball_height")
        self.count = 0
        self.total_height = 0

    def reset(self):
        self.count = 0
        self.total_height = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        ball_z = gamestates[:, StateConstants.BALL_POSITION.start + 2]

        self.total_height += np.sum(ball_z)
        self.count += ball_z.size

    def get_stat(self):
        return self.total_height / (self.count or 1)


class AverageGoalSpeed(StatTracker):
    def __init__(self):
        super().__init__("average_goal_speed")
        self.count = 0
        self.total_speed = 0

    def reset(self):
        self.count = 0
        self.total_speed = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        orange_diff = np.diff(gamestates[:, StateConstants.ORANGE_SCORE], append=np.nan)
        blue_diff = np.diff(gamestates[:, StateConstants.BLUE_SCORE], append=np.nan)

        goal_frames = (orange_diff > 0) | (blue_diff > 0)

        goal_speed = gamestates[goal_frames, StateConstants.BALL_LINEAR_VELOCITY]
        goal_speed = np.linalg.norm(goal_speed, axis=-1)

        self.total_speed += goal_speed.sum() / 27.78  # convert to km/h
        self.count += goal_speed.size

    def get_stat(self):
        return self.total_speed / (self.count or 1)


class TotalSteps(StatTracker):
    def __init__(self):
        super().__init__("total_steps")
        self.count = 0

    def reset(self):
        self.count = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        players = gamestates[:, StateConstants.PLAYERS]  # Shape: (n_ticks, n_players * n_features)
        car_ids = players[:, StateConstants.CAR_IDS]  # Shape: (n_ticks, n_players)
        self.count += car_ids.size

    def get_stat(self):
        return self.count


class GamemodeSpecificTracker(StatTracker):
    def __init__(self, tracker: StatTracker, gamemode: str):
        self.gamemode = tuple(sorted(int(x) for x in gamemode.split("v")))
        gamemode = "v".join(str(x) for x in self.gamemode)
        self.tracker = tracker
        super().__init__(f"{gamemode}/{tracker.name}")

    def reset(self):
        self.tracker.reset()

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        players = gamestates[0, StateConstants.PLAYERS]
        teams = players[0, StateConstants.TEAM_NUMS]
        if (teams == 0).sum() in self.gamemode and (teams == 1).sum() in self.gamemode:
            self.tracker.update(gamestates, mask)

    def get_stat(self):
        return self.tracker.get_stat()
