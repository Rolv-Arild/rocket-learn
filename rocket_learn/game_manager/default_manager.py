from typing import Dict, Literal, Any

import numpy as np
from pettingzoo import ParallelEnv
from rlgym.utils.gamestates import GameState

from rocket_learn.agent.agent import Agent
from rocket_learn.agent.policy import Policy
from rocket_learn.agent.pretrained_policy import HardcodedAgent
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.game_manager.game_manager import GameManager


class DefaultManager(GameManager):
    def __init__(self,
                 env: ParallelEnv,
                 gamemode_weights: dict[str, float],
                 display: Literal[None, "stochastic", "deterministic", "rollout"] = None):
        super(DefaultManager, self).__init__(env)
        self.display = display

        self.gamemode_exp = {k: 0 for k in gamemode_weights.keys()}
        self.gamemode_weights = gamemode_weights

    def generate_matchup(self) -> (Dict[str, Agent], int):
        raise NotImplementedError

    @staticmethod
    def _infer(car_identifer: Dict[str, str], identifier_agent: Dict[str, Agent],
               observations: Dict[str, Any], info: Dict[str, Any]):
        all_actions = {}

        # identifier_model = {}
        identifier_cars = {}
        for car, identifier in car_identifer.items():
            identifier_cars.setdefault(identifier, []).append(car)
            # identifier_model[identifier] = model

        for identifier, cars in identifier_cars.items():
            agent = identifier_agent[identifier]

            if agent.is_multi:
                actions = agent.act({car: observations[car] for car in cars})
            else:
                actions = [agent.act({car: observations[car]}) for car in cars]

            all_actions = dict(zip(cars, actions))

        return all_actions

    def _episode(self, agent_policy: Dict[str, Agent]):
        obs, info = self.env.reset(return_info=True)
        all_states = [self.env.state()]

        while True:
            all_actions = self._infer(agent_policy, obs, info)

            observation, reward, terminated, truncated, info = self.env.step(all_actions)

            all_states.append(self.env.state())

            if any(terminated.values()) or any(truncated.values()):
                break

        return all_states

    def rollout(self, car_agent: Dict[str, Agent]) -> tuple[list[np.ndarray], dict]:
        states, buffers = self._episode(agent_policy)

        steps = sum(b.size() for b in buffers.values())
        mode = self._get_gamemode(agent_policy)
        self.gamemode_exp[mode] += steps

        return states, buffers

    def evaluate(self, agent_policy: Dict[str, Agent]) -> int:
        pass  # TODO

    def show(self, agent_policy: Dict[str, Agent]):
        states, buffers = self._episode(agent_policy)

        steps = len(states)
        mode = self._get_gamemode(agent_policy)
        self.gamemode_exp[mode] += steps

    @staticmethod
    def _get_gamemode(agent_policy: Dict[str, Agent]):
        b = o = 0
        for key in agent_policy.keys():
            if key.startswith("blue-"):
                b += int(key.replace("blue-", ""))
            else:
                o += int(key.replace("orange-", ""))
        mode = f"{min(b, o)}v{max(b, o)}"
        return mode
