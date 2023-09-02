from typing import List

from rlgym.api import AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import GOAL_HEIGHT

from rocket_learn.custom_objects.scoreboard.util import RAMP_RADIUS
from rocket_learn.stat_trackers.stat_tracker import StatTracker


class CarOnGround(StatTracker):
    def __init__(self):
        super().__init__("Percent on ground", "agg_ratio")

    def collect(self, agent_ids: List[AgentID], game_states: List[GameState]):
        total_ground = 0
        total_steps = 0

        for game_state in game_states:
            cars = game_state.cars
            for agent_id in agent_ids:
                car = cars[agent_id]
                if car.on_ground:
                    total_ground += 1
                total_steps += 1

        return 100 * total_ground, total_steps


class CarLowInAir(StatTracker):
    def __init__(self):
        super().__init__("Percent low in air", "agg_ratio")

    def collect(self, agent_ids: List[AgentID], game_states: List[GameState]):
        total_ground = 0
        total_steps = 0

        for game_state in game_states:
            cars = game_state.cars
            for agent_id in agent_ids:
                car = cars[agent_id]
                if not car.on_ground and car.physics.position[2] > GOAL_HEIGHT:
                    total_ground += 1
                total_steps += 1

        return 100 * total_ground, total_steps


class CarOnWall(StatTracker):
    def __init__(self):
        super().__init__("Percent on wall", "agg_ratio")

    def collect(self, agent_ids: List[AgentID], game_states: List[GameState]):
        total_ground = 0
        total_steps = 0

        for game_state in game_states:
            cars = game_state.cars
            for agent_id in agent_ids:
                car = cars[agent_id]
                if car.on_ground and car.physics.position[2] > RAMP_RADIUS:
                    total_ground += 1
                total_steps += 1

        return 100 * total_ground, total_steps


class CarHighInAir(StatTracker):
    def __init__(self):
        super().__init__("Percent high in air", "agg_ratio")

    def collect(self, agent_ids: List[AgentID], game_states: List[GameState]):
        total_ground = 0
        total_steps = 0

        for game_state in game_states:
            cars = game_state.cars
            for agent_id in agent_ids:
                car = cars[agent_id]
                if not car.on_ground and car.physics.position[2] <= GOAL_HEIGHT:
                    total_ground += 1
                total_steps += 1

        return 100 * total_ground, total_steps
