from abc import ABC, abstractmethod
from typing import Tuple
from torch import nn

import pygame
import keyboard

from rocket_learn.agent.discrete_policy import DiscretePolicy

from rlgym.utils.gamestates import GameState

class HardcodedAgent(ABC):
    """
        An external bot prebuilt and imported to be trained against
    """

    @abstractmethod
    def act(self, state: GameState): raise NotImplementedError


class PretrainedDiscretePolicy(DiscretePolicy, HardcodedAgent):
    """
        A rocket-learn discrete policy pretrained and imported to be trained against

        :param obs_builder_func: Function that will generate the correct observation from the gamestate
        :param net: policy net
        :param shape: action distribution shape
    """

    def __init__(self, obs_builder_func, net: nn.Module, shape: Tuple[int, ...] = (3,) * 5 + (2,) * 3):
        super().__init__(net, shape)
        self.obs_builder_func = obs_builder_func

    def act(self, state: GameState):
        obs = self.obs_builder_func(state)
        dist = policy.get_action_distribution(obs)
        action_indices = policy.sample_action(dist, deterministic=False)
        actions = policy.env_compatible(action_indices)

        return actions


class DemoDriveAgent(HardcodedAgent):
    def act(self, state: GameState):
        return [2, 1, 1, 0, 0, 0, 0, 0]


class DemoKBMDriveAgent(HardcodedAgent):
    def act(self, state: GameState):
        return [2, 1, 0, 0, 0]


class HumanAgent(HardcodedAgent):
    def __init__(self):
        pygame.init()
        self.controller_map = {}

        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()

    def controller_actions(self, state):
        player = [p for p in state.players if p.team_num == 0][0]
        # allow controller to activate
        pygame.event.pump()

        jump = self.joystick.get_button(0)
        boost = self.joystick.get_button(1)
        handbrake = self.joystick.get_button(2)

        throttle = self.joystick.get_axis(5)
        throttle = max(0, throttle)

        reverse_throttle = self.joystick.get_axis(4)
        reverse_throttle = max(0, reverse_throttle)

        throttle = throttle - reverse_throttle
        throttle += 1

        steer = self.joystick.get_axis(0)
        steer += 1
        if abs(1 - steer) < .2:
            steer = 1

        pitch = self.joystick.get_axis(1)
        pitch += 1
        if abs(1 - pitch) < .2:
            pitch = 1

        yaw = steer

        roll = 1
        roll_button = self.joystick.get_button(4)
        if roll_button or jump:
            roll = steer

        return [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]


    def kbm_actions(self, state):
        player = [p for p in state.players if p.team_num == 0][0]

        throttle = 1
        if keyboard.is_pressed('w'):
            throttle = 2
        if keyboard.is_pressed('s'):
            throttle = 0

        steer = 1
        if keyboard.is_pressed('d'):
            steer = 2
        if keyboard.is_pressed('a'):
            steer = 0

        pitch = -throttle

        yaw = steer

        roll = 1
        if keyboard.is_pressed('e'):
            roll = 2
        if keyboard.is_pressed('q'):
            roll = 0

        jump = 0
        if keyboard.is_pressed('f'):
            jump = 1

        boost = 0
        handbrake = 0

        return [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]

    def act(self, state: GameState):
        #actions = self.controller_actions(state)
        actions = self.kbm_actions(state)

        return actions
