import os
import wandb
import numpy
from typing import Any

import torch.jit
from torch.nn import Linear, Sequential

from redis import Redis

from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions.default_reward import DefaultReward
from rlgym.utils.action_parsers.discrete_act import DiscreteAction

from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.utils.util import SplitLayer


# rocket-learn always expects a batch dimension in the built observation
class ExpandAdvancedObs(AdvancedObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)


if __name__ == "__main__":
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(project="demo", entity="wandb_username")
    logger.name = "LOADING_RUN_EXAMPLE"

    redis = Redis(password="you_better_use_a_password")


    def obs():
        return ExpandAdvancedObs()

    def rew():
        return DefaultReward()

    def act():
        return DiscreteAction()


    # -clear DELETE REDIS ENTRIES WHEN STARTING UP (SET TO FALSE TO CONTINUE WITH OLD AGENTS)
    rollout_gen = RedisRolloutGenerator(redis, obs, rew, act,
                                        logger=logger, 
                                        save_every=100,
                                        clear=False)

    critic = Sequential(Linear(107, 128), Linear(128, 64), Linear(64, 32), Linear(32, 1))
    actor = DiscretePolicy(
        Sequential(Linear(107, 128), Linear(128, 64), Linear(64, 32), Linear(32, 21), SplitLayer()))

    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": 5e-5},
        {"params": critic.parameters(), "lr": 5e-5}
    ])

    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)

    alg = PPO(
        rollout_gen,
        agent,
        ent_coef=0.01,
        n_steps=1_000_000,
        batch_size=20_000,
        minibatch_size=10_000,
        epochs=10,
        gamma=599 / 600,
        logger=logger,
    )


    # LOAD A CHECKPOINT THAT WAS PREVIOUSLY SAVED AND CONTINUE TRAINING. OPTIONAL PARAMETER ALLOWS YOU
    # TO RESTART THE STEP COUNT INSTEAD OF CONTINUING
    alg.load("path\\from\\below\\checkpoint.pt")


    # OPTIONAL: FOR A PRETRAINED NETWORK, FREEZE THE POLICY NETWORK TO ALLOW THE CRITIC TO SETTLE
    # commented out here to keep you from accidentally adding it via copy/paste
    # alg.freeze_policy(frozen_iterations=200)

    
    # BEGIN TRAINING. IT WILL CONTINUE UNTIL MANUALLY STOPPED
    # -iterations_per_save SPECIFIES HOW OFTEN CHECKPOINTS ARE SAVED
    # -save_dir SPECIFIES WHERE
    alg.run(iterations_per_save=100, save_dir="checkpoint_save_directory")
