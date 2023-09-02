from typing import List

from rlgym.api import AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import GOAL_HEIGHT

from rocket_learn.custom_objects.scoreboard.util import SECONDS_PER_MINUTE, TICKS_PER_SECOND, RAMP_RADIUS
from rocket_learn.stat_trackers.stat_tracker import StatTracker


class Touch(StatTracker):
    def __init__(self):
        super().__init__("Touches per minute", "agg_ratio")

    def collect(self, agent_ids: List[AgentID], game_states: List[GameState]):
        total_touches = 0
        total_ticks = 0

        for game_state in game_states:
            cars = game_state.cars
            for agent_id in agent_ids:
                car = cars[agent_id]
                total_touches += car.ball_touches
        dt = game_states[-1].tick_count - game_states[0].tick_count
        total_ticks += dt * len(agent_ids)

        return (SECONDS_PER_MINUTE * TICKS_PER_SECOND * total_touches,
                total_ticks)


class TouchHeight(StatTracker):
    def __init__(self):
        super().__init__("Average touch height", "agg_ratio")

    def collect(self, agent_ids: List[AgentID], game_states: List[GameState]):
        total_touches = 0
        total_height = 0

        for game_state in game_states:
            cars = game_state.cars
            for agent_id in agent_ids:
                car = cars[agent_id]
                if car.ball_touches > 0:
                    ball_z = game_state.ball.position[2]
                    total_height += car.ball_touches * ball_z
                    total_touches += car.ball_touches

        return total_height, total_touches


class LowAirTouch(StatTracker):
    def __init__(self):
        super().__init__("Low air touches per minute", "agg_ratio")

    def collect(self, agent_ids: List[AgentID], game_states: List[GameState]):
        total_touches = 0
        total_ticks = 0

        for game_state in game_states:
            cars = game_state.cars
            for agent_id in agent_ids:
                car = cars[agent_id]
                if car.ball_touches > 0 \
                        and not car.on_ground \
                        and game_state.ball.position[2] <= GOAL_HEIGHT:
                    total_touches += car.ball_touches
        dt = game_states[-1].tick_count - game_states[0].tick_count
        total_ticks += dt * len(agent_ids)

        return (SECONDS_PER_MINUTE * TICKS_PER_SECOND * total_touches, \
                total_ticks)


class WallTouch(StatTracker):
    def __init__(self):
        super().__init__("Wall touches per minute", "agg_ratio")

    def collect(self, agent_ids: List[AgentID], game_states: List[GameState]):
        total_touches = 0
        total_ticks = 0

        for game_state in game_states:
            cars = game_state.cars
            for agent_id in agent_ids:
                car = cars[agent_id]
                if car.ball_touches > 0 \
                        and car.on_ground \
                        and car.ball_touches > RAMP_RADIUS:
                    total_touches += car.ball_touches
        dt = game_states[-1].tick_count - game_states[0].tick_count
        total_ticks += dt * len(agent_ids)

        return (SECONDS_PER_MINUTE * TICKS_PER_SECOND * total_touches,
                total_ticks)


class HighAirTouch(StatTracker):
    def __init__(self):
        super().__init__("High air touches per minute", "agg_ratio")

    def collect(self, agent_ids: List[AgentID], game_states: List[GameState]):
        total_touches = 0
        total_ticks = 0

        for game_state in game_states:
            cars = game_state.cars
            for agent_id in agent_ids:
                car = cars[agent_id]
                if car.ball_touches > 0 \
                        and not car.on_ground \
                        and game_state.ball.position[2] > GOAL_HEIGHT:
                    total_touches += car.ball_touches
        dt = game_states[-1].tick_count - game_states[0].tick_count
        total_ticks += dt * len(agent_ids)

        return (SECONDS_PER_MINUTE * TICKS_PER_SECOND * total_touches,
                total_ticks)
