import numpy as np

from rocket_learn.utils.gamestate_encoding import StateConstants
from rocket_learn.utils.stat_trackers.stat_tracker import StatTracker


class Speed(StatTracker):
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


class Demos(StatTracker):
    def __init__(self):
        super().__init__("average_demos")
        self.count = 0
        self.total_demos = 0

    def reset(self):
        self.count = 0
        self.total_demos = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        players = gamestates[:, StateConstants.PLAYERS]

        demos = players[-1, StateConstants.MATCH_DEMOLISHES] - players[0, StateConstants.MATCH_DEMOLISHES]
        self.total_demos += np.sum(demos)
        self.count += demos.size

    def get_stat(self):
        return self.total_demos / (self.count or 1)


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


class Touch(StatTracker):
    def __init__(self):
        super().__init__("touch_rate")
        self.count = 0
        self.total_touches = 0

    def reset(self):
        self.count = 0
        self.total_touches = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        players = gamestates[:, StateConstants.PLAYERS]
        is_touch = players[:, StateConstants.BALL_TOUCHED]

        self.total_touches += np.sum(is_touch)
        self.count += is_touch.size

    def get_stat(self):
        return self.total_touches / (self.count or 1)


class EpisodeLength(StatTracker):
    def __init__(self):
        super().__init__("episode_length")
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


class Boost(StatTracker):
    def __init__(self):
        super().__init__("average_boost")
        self.count = 0
        self.total_boost = 0

    def reset(self):
        self.count = 0
        self.total_boost = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        players = gamestates[:, StateConstants.PLAYERS]
        boost = np.clip(players[:, StateConstants.BOOST_AMOUNT], 0, 1)
        self.total_boost += np.sum(boost)
        self.count += boost.size

    def get_stat(self):
        return self.total_boost / (self.count or 1)


class BehindBall(StatTracker):
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
        behind = (2 * players[:, StateConstants.TEAM_NUMS] - 1) * (ball_y.reshape(-1, 1) - player_y) < 0

        self.total_behind += np.sum(behind)
        self.count += behind.size

    def get_stat(self):
        return self.total_behind / (self.count or 1)


class TouchHeight(StatTracker):
    def __init__(self):
        super().__init__("touch_height")
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


class DistToBall(StatTracker):
    def __init__(self):
        super().__init__("distance_to_ball")
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


class AirTouchHeight(StatTracker):
    def __init__(self):
        super().__init__("air_touch_height")
        self.count = 0
        self.total_height = 0

    def reset(self):
        self.count = 0
        self.total_height = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        players = gamestates[:, StateConstants.PLAYERS]
        ball_z = gamestates[:, StateConstants.BALL_POSITION.start + 2]
        touch_heights = ball_z[players[:, StateConstants.BALL_TOUCHED].any(axis=1)]
        touch_heights = touch_heights[touch_heights >= 175]  # remove dribble touches and below

        self.total_height += np.sum(touch_heights)
        self.count += touch_heights.size

    def get_stat(self):
        return self.total_height / (self.count or 1)


class BallSpeed(StatTracker):
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


class BallHeight(StatTracker):
    def __init__(self):
        super().__init__("ball_height")
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


class GoalSpeed(StatTracker):
    def __init__(self):
        super().__init__("avg_goal_speed")
        self.count = 0
        self.total_speed = 0

    def reset(self):
        self.count = 0
        self.total_speed = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        ball_speeds = gamestates[:, StateConstants.BALL_LINEAR_VELOCITY]
        goal_speed = ball_speeds[-1]
        goal_speed = np.linalg.norm(goal_speed)

        self.total_speed += goal_speed
        self.count += 1

    def get_stat(self):
        return self.total_speed / (self.count or 1)


class MaxGoalSpeed(StatTracker):
    def __init__(self):
        super().__init__("max_goal_speed")
        self.max_speed = 0

    def reset(self):
        self.max_speed = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        ball_speeds = gamestates[:, StateConstants.BALL_LINEAR_VELOCITY]
        goal_speed_array = ball_speeds[-1]
        goal_speed = float(np.linalg.norm(goal_speed_array))

        self.max_speed = max(float(self.max_speed), goal_speed)

    def get_stat(self):
        return self.max_speed


class CarOnGround(StatTracker):
    def __init__(self):
        super().__init__("pct_car_on_ground")
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
