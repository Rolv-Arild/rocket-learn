from typing import Any
import numpy

from redis import Redis

from rlgym.envs import Match
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym.utils.reward_functions.default_reward import DefaultReward
from rlgym.utils.state_setters.default_state import DefaultState
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym.utils.action_parsers.discrete_act import DiscreteAction

from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutWorker
from rocket_learn.agent.pretrained_agents.human_agent import HumanAgent


# ROCKET-LEARN ALWAYS EXPECTS A BATCH DIMENSION IN THE BUILT OBSERVATION
class ExpandAdvancedObs(AdvancedObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)

"""

Allows the worker to run a human player, letting the AI play against and learn from human interation.

Important things to note:

-The human will always be blue due to RLGym camera constraints
-Attempting to run a human trainer and pretrained agents will cause the pretrained agents to be ignored. 
    They will never show up.

"""

if __name__ == "__main__":
    match = Match(
        game_speed=1,
        self_play=True,
        team_size=1,
        state_setter=DefaultState(),
        obs_builder=ExpandAdvancedObs(),
        action_parser=DiscreteAction(),
        terminal_conditions=[TimeoutCondition(round(2000)),
                             GoalScoredCondition()],
        reward_function=DefaultReward()
    )

    # ALLOW HUMAN CONTROL THROUGH MOUSE AND KEYBOARD OR A CONTROLLER IF ONE IS PLUGGED IN
    # -CONTROL BINDINGS ARE CURRENTLY NOT CHANGEABLE
    # -CONTROLLER SETUP CURRENTLY EXPECTS AN XBOX 360 CONTROLLER. OTHERS WILL WORK BUT PROBABLY NOT WELL
    human = HumanAgent()

    r = Redis(host="127.0.0.1", password="you_better_use_a_password")

    # LAUNCH ROCKET LEAGUE AND BEGIN TRAINING
    # -human_agent TELLS RLGYM THAT THE FIRST AGENT IS ALWAYS TO BE HUMAN CONTROLLED
    # -past_version_prob SPECIFIES HOW OFTEN OLD VERSIONS WILL BE RANDOMLY SELECTED AND TRAINED AGAINST
    RedisRolloutWorker(r, "exampleHumanWorker", match, human_agent=human, past_version_prob=.05).run()
