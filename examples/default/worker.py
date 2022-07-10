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

from rocket_learn.rollout_generator.redis.redis_rollout_worker import RedisRolloutWorker


# ROCKET-LEARN ALWAYS EXPECTS A BATCH DIMENSION IN THE BUILT OBSERVATION
class ExpandAdvancedObs(AdvancedObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)


if __name__ == "__main__":
    """

    Starts up a rocket-learn worker process, which plays out a game, sends back game data to the 
    learner, and receives updated model parameters when available

    """

    # OPTIONAL ADDITION:
    # LIMIT TORCH THREADS TO 1 ON THE WORKERS TO LIMIT TOTAL RESOURCE USAGE
    # TRY WITH AND WITHOUT FOR YOUR SPECIFIC HARDWARE
    import torch

    torch.set_num_threads(1)

    # BUILD THE ROCKET LEAGUE MATCH THAT WILL USED FOR TRAINING
    # -ENSURE OBSERVATION, REWARD, AND ACTION CHOICES ARE THE SAME IN THE WORKER
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

    # LINK TO THE REDIS SERVER YOU SHOULD HAVE RUNNING (USE THE SAME PASSWORD YOU SET IN THE REDIS
    # CONFIG)
    r = Redis(host="127.0.0.1", password="you_better_use_a_password")

    # LAUNCH ROCKET LEAGUE AND BEGIN TRAINING
    # -past_version_prob SPECIFIES HOW OFTEN OLD VERSIONS WILL BE RANDOMLY SELECTED AND TRAINED AGAINST
    RedisRolloutWorker(r, "example", match,
                       past_version_prob=.2,
                       evaluation_prob=0.01,
                       sigma_target=2,
                       dynamic_gm=False,
                       send_obs=True,
                       streamer_mode=False,
                       send_gamestates=False,
                       force_paging=False,
                       auto_minimize=True,
                       local_cache_name="example_model_database").run()
