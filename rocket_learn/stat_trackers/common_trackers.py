from typing import List

import numpy as np
from rlgym.api import AgentID
from rlgym.rocket_league.api import GameState

from rocket_learn.custom_objects.scoreboard.util import TICKS_PER_SECOND, SECONDS_PER_MINUTE
from rocket_learn.stat_trackers.stat_tracker import StatTracker


class Speed(StatTracker):
    def __init__(self):
        super().__init__("Average speed (uu/s)", "agg_ratio")

    def collect(self, agent_ids: List[AgentID], game_states: List[GameState]):
        total_speed = 0.0
        total_steps = 0

        for game_state in game_states:
            cars = game_state.cars
            for agent_id in agent_ids:
                car = cars[agent_id]
                speed = np.linalg.norm(car.physics.linear_velocity)
                total_speed += speed
                total_steps += 1

        return total_speed, total_steps


class Demos(StatTracker):
    def __init__(self):
        super().__init__("Demos per minute", "agg_ratio")

    def collect(self, agent_ids: List[AgentID], game_states: List[GameState]):
        total_demos = 0
        total_ticks = 0

        for game_state in game_states:
            cars = game_state.cars
            for agent_id in agent_ids:
                car = cars[agent_id]
                victim = car.bump_victim_id
                if victim is not None and cars[victim].demo_respawn_timer > 0:
                    total_demos += 1
        dt = game_states[-1].tick_count - game_states[0].tick_count
        total_ticks += dt * len(agent_ids)

        return (SECONDS_PER_MINUTE * TICKS_PER_SECOND * total_demos,
                total_ticks)


class Boost(StatTracker):
    def __init__(self):
        super().__init__("Average boost", "agg_ratio")

    def collect(self, agent_ids: List[AgentID], game_states: List[GameState]):
        total_boost = 0
        total_steps = 0

        for game_state in game_states:
            cars = game_state.cars
            for agent_id in agent_ids:
                car = cars[agent_id]
                total_boost += car.boost_amount
                total_steps += 1

        return total_boost, total_steps
