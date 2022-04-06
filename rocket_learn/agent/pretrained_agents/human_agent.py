import pygame
import keyboard

from rlgym.utils.gamestates import GameState

from rocket_learn.agent.pretrained_policy import HardcodedAgent

class HumanAgent(HardcodedAgent):
    def __init__(self):
        pygame.init()
        self.controller_map = {}

        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print("Controller found")

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

        steer = self.joystick.get_axis(0)
        if abs(steer) < .2:
            steer = 0

        pitch = self.joystick.get_axis(1)
        if abs(pitch) < .2:
            pitch = 0

        yaw = steer

        roll = 0
        roll_button = self.joystick.get_button(4)
        if roll_button or jump:
            roll = steer

        return [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]


    def kbm_actions(self, state):
        player = [p for p in state.players if p.team_num == 0][0]

        throttle = 0
        if keyboard.is_pressed('w'):
            throttle = 1
        if keyboard.is_pressed('s'):
            throttle = -1

        steer = 0
        if keyboard.is_pressed('d'):
            steer = 1
        if keyboard.is_pressed('a'):
            steer = -1

        pitch = -throttle

        yaw = steer

        roll = 0
        if keyboard.is_pressed('e'):
            roll = 1
        if keyboard.is_pressed('q'):
            roll = -1

        jump = 0
        if keyboard.is_pressed('f'):
            jump = 1

        boost = 0
        handbrake = 0

        return [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]

    def act(self, state: GameState, player_index: int):
        if self.joystick:
            actions = self.controller_actions(state)
        else:
            actions = self.kbm_actions(state)

        return actions
