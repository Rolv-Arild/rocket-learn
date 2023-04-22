from typing import Dict, Literal, Tuple, List, Iterable

import numpy as np

from rocket_learn.agent.agent import Agent
from rocket_learn.envs.rocket_league import RocketLeague
from rocket_learn.game_manager.game_manager import GameManager

DefaultMatchup = Tuple[Dict[str, str], Dict[str, Agent]]


class DefaultManager(GameManager):
    def __init__(self,
                 env: RocketLeague,
                 gamemode_weights: dict[str, float],
                 display: Literal[None, "stochastic", "deterministic", "rollout"] = None):
        super(DefaultManager, self).__init__(env)
        self.display = display

        self.gamemode_exp = {k: 0 for k in gamemode_weights.keys()}
        self.gamemode_weights = gamemode_weights

    def generate_matchup(self) -> Tuple[DefaultMatchup, int]:
        raise NotImplementedError

    def _episode(self, car_identifier, identifier_agent, boost_consumption=1, gravity=1):
        options = dict(boost_consumption=boost_consumption, gravity=gravity,
                       blue=sum(k.startswith("blue") for k in car_identifier),
                       orange=sum(k.startswith("orange") for k in car_identifier))
        observations, info = self.env.reset(options=options)
        all_states = [self.env.state()]

        identifier_cars = {}
        for car, identifier in car_identifier.items():
            identifier_cars.setdefault(identifier, set()).add(car)

        total_agent_steps = 0

        removed_cars = set()
        while True:
            # Select actions
            all_actions = {}
            for identifier, cars in identifier_cars.items():
                agent = identifier_agent[identifier]

                actions = agent.act({car: observations[car] for car in cars})

                all_actions.update(actions)

            # Step
            observations, rewards, terminated, truncated, info = self.env.step(all_actions)
            total_agent_steps *= len(self.env.agents)

            state = self.env.state()
            all_states.append(state)

            # Remove agents that are terminated or truncated
            for car in self.env.agents:
                ended = {}
                if terminated[car] or truncated[car]:
                    identifier = car_identifier[car]
                    ended.setdefault(identifier, {}).update({car: truncated[car]})
                    removed_cars.add(car)

                for identifier, truncs in ended.items():
                    agent = identifier_agent[identifier]
                    agent.end(state, truncs)

            # End if there are no agents left
            if not self.env.agents:
                break

        return all_states, total_agent_steps

    def rollout(self, matchup: DefaultMatchup):
        car_identifier, identifier_agent = matchup
        states, steps = self._episode(car_identifier, identifier_agent)

        steps = len(states) * len(states[0])
        mode = self._get_gamemode(car_identifier.keys())
        self.gamemode_exp[mode] += steps

    def evaluate(self, matchup: DefaultMatchup) -> int:
        # TODO, how do we handle scoreboard?
        #  Maybe include it by default in RLGym obs?
        #  Need to handle updating, state setting, terminal and reward still
        #  Could include in options maybe, but ideally StateSetter should do it I guess
        #  Default is clearly overtime (doesn't end unless goal is scored)
        pass

    def show(self, matchup: DefaultMatchup):
        states, steps = self._episode(*matchup)

        steps = len(states)
        mode = self._get_gamemode(matchup[0].keys())
        self.gamemode_exp[mode] += steps

    @staticmethod
    def _get_gamemode(cars: Iterable[str]):
        b = o = 0
        for key in cars:
            if key.startswith("blue-"):
                b += int(key.replace("blue-", ""))
            else:
                o += int(key.replace("orange-", ""))
        mode = f"{min(b, o)}v{max(b, o)}"
        return mode
