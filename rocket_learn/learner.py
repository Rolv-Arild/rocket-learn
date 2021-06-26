import os
import pickle
import time

import numpy as np

from redis import Redis
import msgpack

import torch
import torch.nn as nn

import multiprocessing as mp


#example pytorch stuff, delete later
actor = nn.Sequential(
    nn.Linear(state_dim, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, action_dim),
    nn.Softmax(dim=-1)
)

# critic
critic = nn.Sequential(
    nn.Linear(state_dim, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
)

class learner:
    def __init__(self):
        self.algorithm = PPO(actor, critic)

        #**DEFAULT NEEDS TO INCORPORATE BASIC SECURITY, THIS IS NOT SUFFICIENT**
        self.redis = Redis(host='127.0.0.1', port=6379)

        # <-- build workers either here or externally to avoid linkage
        # should we use rolv's sb3 code or can we do better not being tied to sb3?

        #might be better to move this "main work step" to an external class
        while True:
            rollouts = self.recieve_worker_data()
            self.calculate(rollouts)
            #send updated policy

    def recieve_worker_data(self):
        while True:
            item = self.redis.lpop(ROLLOUTS)
            if item is not None:
                rollout = msgpack.loads(item)
                yield rollout
            else:
                time.sleep(10)

    def calculate(self):
        #apply PPO now but separate so we can refactor to allow different algorithm types
        self.algorithm.calculate()


#this should probably be in its own file
class PPO:
    def __init__(self, actor, critic, n_rollouts = 36, lr_actor = 3e-4, lr_critic = 3e-4, gamma = 0.9, epochs = 1):
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer

        self.epochs = epochs

        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr_actor},
            {'params': self.critic.parameters(), 'lr': lr_critic}
        ])

    def calculate(self):
        for e in range(self.epochs):
            #PPO MATH HERE, DIFFERENT WAYS TO DO THIS, DO WE CARE WHICH IMPLEMENTATION?
            pass


