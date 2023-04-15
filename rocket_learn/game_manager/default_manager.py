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
    def __init__(self, env: ParallelEnv, gamemode_weights,
                 display: Literal[None, "stochastic", "deterministic", "rollout"] = None):
        super(DefaultManager, self).__init__(env)
        self.display = display

        self.gamemode_exp = {k: 0 for k in gamemode_weights.keys()}
        self.gamemode_weights = gamemode_weights

    def generate_matchup(self) -> (Dict[str, Agent], int):
        raise NotImplementedError

    @staticmethod
    def _infer(agent_policy: Dict[str, Agent], observations: Dict[str, Any], info: Dict[str, Any]):
        all_action_indices = {}
        all_log_probs = {}
        all_actions = {}

        identifier_model = {}
        identifier_cars = {}
        for car, model in agent_policy.items():
            identifier_cars.setdefault(model.identifier, []).append(car)
            identifier_model[model.identifier] = model

        for identifier, cars in identifier_cars.items():
            model = identifier_model[identifier]
            if isinstance(model, Policy):
                obs = []
                for car in cars:
                    o = observations[car]
                    obs.append(o)
                if isinstance(obs[0], tuple):
                    obs = tuple(np.stack(o) for o in zip(*obs))
                else:
                    obs = np.stack(obs)
                dist = model.get_action_distribution(obs)
                action_indices = model.sample_action(dist).numpy()
                log_probs = model.log_prob(dist, action_indices).numpy()
                actions = model.env_compatible(action_indices)

                if model.identifier < 0:  # Maybe different check for latest?
                    for i, car in enumerate(cars):
                        all_action_indices[car] = action_indices[i]
                        all_log_probs[car] = log_probs[i]
                        all_actions[car] = actions[i]
            elif isinstance(model, HardcodedAgent):
                for car in cars:
                    if car.startswith("blue-"):
                        idx = int(car.replace("blue-", ""))
                    else:
                        idx = max(int(c.replace("blue-", "")) for c in agent_policy.keys())
                        idx += 1 + int(car.replace("orange-", ""))
                    actions = model.act(info[car]["state"], idx)
                    all_actions[car] = actions
            else:
                print(str(type(model)) + " type use not defined")
                assert False

        return all_actions, all_action_indices, all_log_probs

    def _episode(self, agent_policy: Dict[str, Agent]):
        buffers = {a: ExperienceBuffer() for a in agent_policy.keys()}

        obs, info = self.env.reset(return_info=True)
        all_states = [self.env.state()]

        while True:
            all_actions, all_action_indices, all_log_probs = \
                self._infer(agent_policy, obs, info)

            observation, reward, terminated, truncated, info = self.env.step(all_actions)

            for car in all_action_indices.keys():
                buffers[car].add_step(observation[car], all_action_indices[car], reward[car],
                                      terminated[car], truncated[car], all_log_probs[car], info[car])

            all_states.append(self.env.state())

            if any(terminated.values()) or any(truncated.values()):
                break

        return all_states, buffers

    def rollout(self, agent_policy: Dict[str, Agent]) -> tuple[list[np.ndarray], dict]:
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
