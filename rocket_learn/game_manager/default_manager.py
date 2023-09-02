from typing import Dict, Literal, Tuple, List, Optional

from rlgym.api import RLGym, AgentID
from rlgym.rocket_league.api import GameState

from rocket_learn.agent.agent import Agent
from rocket_learn.custom_objects.custom_object_logic import CustomObjectLogic
from rocket_learn.custom_objects.scoreboard.util import TICKS_PER_SECOND
from rocket_learn.game_manager.game_manager import GameManager

DefaultMatchup = List[Tuple[Agent, List[AgentID]]]


class DefaultManager(GameManager):
    def __init__(self,
                 envs: Dict[int, RLGym],
                 gamemode_weights: Dict[str, float],
                 custom_objects: Optional[List[CustomObjectLogic]] = None,
                 display: Literal[None, "stochastic", "deterministic", "rollout"] = None,
                 ):
        super(DefaultManager, self).__init__(envs)
        if custom_objects is None:
            custom_objects = []
        self.display = display

        self.gamemode_exp = {k: 0 for k in gamemode_weights.keys()}
        self.gamemode_weights = gamemode_weights
        self.custom_objects = custom_objects

    def generate_matchup(self) -> Tuple[DefaultMatchup, int]:
        raise NotImplementedError

    @staticmethod
    def _episode(env, matchup: DefaultMatchup):
        observations = env.reset()

        total_agent_steps = 0

        all_states = []
        while True:
            # Select actions
            all_actions = {}
            for agent, agent_ids in matchup:
                actions = agent.act({agent_id: observations[agent_id] for agent_id in agent_ids})

                all_actions.update(actions)
                total_agent_steps += len(agent_ids)

            # Step
            observations, rewards, terminated, truncated = env.step(all_actions)

            state = env.state()
            all_states.append(state)

            if all(terminated.values()) or all(truncated.values()):
                break

        return all_states, total_agent_steps

    def rollout(self, matchup: DefaultMatchup):
        env = self.envs[self.ROLLOUT]
        states, steps = self._episode(env, matchup)

        mode = self._get_gamemode(states[0])
        self.gamemode_exp[mode] += steps

    def evaluate(self, matchup: DefaultMatchup):
        env = self.envs[self.EVAL]

        b = o = 0
        while True:
            self._episode(env, matchup)

            timed_out = (b == env.custom_object_logic.blue and
                         o == env.custom_object_logic.orange)  # No touch/timeout triggered

            b = env.custom_object_logic.blue
            o = env.custom_object_logic.orange
            time_left = env.custom_object_logic.ticks_left / TICKS_PER_SECOND

            term, trunc = env.custom_object_logic.done()
            if term or ff or timed_out:
                break

        env.custom_object_logic = old_custom_object_logic
        env.state_setter = old_state_setter
        env.terminal_conditions = old_terminals

        # TODO, how do we handle scoreboard?
        #  Maybe include it by default in RLGym obs?
        #  Need to handle updating, state setting, terminal and reward still
        #  Could include in options maybe, but ideally StateSetter should do it I guess
        #  Default is clearly overtime (doesn't end unless goal is scored)
        pass

    def show(self, matchup: DefaultMatchup):
        env = self.SHOW
        states, steps = self._episode(env, matchup)

        steps = len(states)
        mode = self._get_gamemode(states[0])
        self.gamemode_exp[mode] += steps

    @staticmethod
    def _get_gamemode(state: GameState):
        b = o = 0
        for car in state.cars.values():
            if car.is_blue:
                b += 1
            else:
                o += 1
        mode = f"{min(b, o)}v{max(b, o)}"
        return mode
