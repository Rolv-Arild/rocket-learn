from typing import List

import numpy as np
from rlgym.api import AgentID
from rlgym.rocket_league.api import GameState

from rocket_learn.stat_trackers.stat_tracker import StatTracker


class BehindBall(StatTracker):
    def __init__(self):
        super().__init__("Percent behind ball", "agg_ratio")

    def collect(self, agent_ids: List[AgentID], game_states: List[GameState]):
        total_behind = 0
        total_steps = 0

        for game_state in game_states:
            cars = game_state.cars
            ball_y = game_state.ball.position[1]
            for agent_id in agent_ids:
                car = cars[agent_id]
                car_y = car.physics.position[1]
                if (car.is_blue and ball_y > car_y
                        or car.is_orange and ball_y < car_y):
                    total_behind += 1
                total_steps += 1

        return 100 * total_behind, total_steps


class DistToBall(StatTracker):
    def __init__(self):
        super().__init__("Average distance to ball", "agg_ratio")

    def collect(self, agent_ids: List[AgentID], game_states: List[GameState]):
        total_dist = 0.0
        total_steps = 0

        for game_state in game_states:
            cars = game_state.cars
            ball_pos = game_state.ball.position
            for agent_id in agent_ids:
                car = cars[agent_id]
                car_pos = car.physics.position
                total_dist += np.linalg.norm(ball_pos - car_pos)
                total_steps += 1

        return total_dist, total_steps
