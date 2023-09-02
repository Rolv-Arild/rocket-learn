from typing import List

import numpy as np
from rlgym.api import AgentID
from rlgym.rocket_league.api import GameState

from rocket_learn.stat_trackers.stat_tracker import StatTracker


class BallHeight(StatTracker):
    def __init__(self):
        super().__init__("Average ball height", "agg_ratio")

    def collect(self, agent_ids: List[AgentID], game_states: List[GameState]):
        total_height = 0.0
        total_steps = 0

        for game_state in game_states:
            total_height += game_state.ball.position[2]
            total_steps += 1

        return total_height, total_steps


class BallSpeed(StatTracker):
    def __init__(self):
        super().__init__("Average ball speed", "agg_ratio")

    def collect(self, agent_ids: List[AgentID], game_states: List[GameState]):
        total_speed = 0.0
        total_steps = 0

        for game_state in game_states:
            speed = np.linalg.norm(game_state.ball.linear_velocity)
            total_speed += speed
            total_steps += 1

        return total_speed, total_steps


class GoalSpeed(StatTracker):
    def __init__(self):
        super().__init__("Average goal speed", "agg_ratio")

    def collect(self, agent_ids: List[AgentID], game_states: List[GameState]):
        total_speed = 0.0
        total_goals = 0

        for game_state in game_states:
            if game_state.goal_scored:
                total_speed += game_state.ball.linear_velocity
                total_goals += 1

        return total_speed, total_goals
