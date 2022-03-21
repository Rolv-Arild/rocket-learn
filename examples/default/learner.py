import os
import wandb

import torch.jit
from torch.nn import Linear, Sequential

from redis import Redis

from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.utils.util import SplitLayer


if __name__ == "__main__":
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(project="demo", entity="example_entity")

    redis = Redis(password="you_better_use_a_password")
    rollout_gen = RedisRolloutGenerator(redis, save_every=10, logger=logger)

    critic = Sequential(Linear(107, 128), Linear(128, 64), Linear(64, 32), Linear(32, 1))
    actor = DiscretePolicy(Sequential(Linear(107, 128), Linear(128, 64), Linear(64, 32), Linear(32, 21), SplitLayer()))

    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": 5e-5},
        {"params": critic.parameters(), "lr": 5e-5}
    ])

    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)

    alg = PPO(
        rollout_gen,
        agent,
        ent_coef=0.01,
        n_steps=1_00_000,
        batch_size=20_000,
        minibatch_size=10_000,
        epochs=10,
        gamma=599 / 600,
        logger=logger,
    )

    alg.run(epochs_per_save=10, save_dir="ppos")
