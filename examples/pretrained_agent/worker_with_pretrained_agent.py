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
from rocket_learn.agent.pretrained_policy import DemoDriveAgent

# rocket-learn always expects a batch dimension in the built observation
class ExpandAdvancedObs(AdvancedObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)

class TotallyDifferentObs(AdvancedObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)



"""

Allows the worker to add in already built agents into training

Important things to note:

-RLGym only accepts 1 action parser. All agents will need to have the same parser or use a combination parser

"""

if __name__ == "__main__":
    match = Match(
        game_speed=100,
        self_play=True,
        team_size=1,
        state_setter=DefaultState(),
        obs_builder=ExpandAdvancedObs(),
        action_parser=DiscreteAction(),
        terminal_conditions=[TimeoutCondition(round(2000)),
                             GoalScoredCondition()],
        reward_function=DefaultReward()
    )

    # TODO: add in pretrained example

    demo_agent = DemoDriveAgent()
    #demo_rl_agent = PretrainedDiscretePolicy(obs=TotallyDifferentObs, net=)

    #agents and their probability of occurrence
    pretrained_agents = {demo_agent: .4} #, demo_rl_agent: .4}

    r = Redis(host="127.0.0.1", password="you_better_use_a_password")
    RedisRolloutWorker(r, "examplePretrainedWorker", match, pretrained_agents=pretrained_agents, past_version_prob=.05).run()
