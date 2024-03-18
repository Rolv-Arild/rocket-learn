# rocket-learn

## What is rocket-learn?

rocket-learn is a machine learning framework specifically designed for Rocket League Reinforcement Learning. It works in conjunction with Rocket League, [RLGym](https://rlgym.org/), and [Bakkesmod](https://bakkesmod.com/).

## What features does rocket-learn have?

- Reinforcement learning algorithm available out of the box
  - Proximal Policy Optimization (PPO)
  - Extensible format allows new algorithms to be added
- Distributed compute from multiple computers
- Automatic saving of and training against previous agent versions
- Trueskill progress tracking
- Training against Hardcoded/Pretrained Agents
- Training against Humans
- Saving and loading models
- wandb logging

## Should I use rocket-learn?

You should use [Stable Baselines3 (SB3)](https://stable-baselines3.readthedocs.io/en/master/) to make your bot at first. The hardest parts of building a machine learning bot are:

- Understanding how to program
- Understanding how machine learning works
- Choosing good hyperparameters
- Choosing good reward functions
- Choosing an action parser
- Making a statesetter that puts the bot in the best situations

SB3 is a great way to figure out those essential parts. Once you have all of those aspects down, rocket-learn may be a good next step to a better machine learning bot.

If you *don't* yet have these, rocket-learn will add a large amount of complexity for no added benefit. It's important to remember that high compute and a tough opponent are less important than good fundamentals of ML.

## How do I setup rocket-learn?

1) Get [Redis](https://docs.servicestack.net/install-redis-windows) running

*__Improper Redis setup can leave your computer extremely vulnerable to Bad Guys. We are not responsible for your computer's safety. We assume you know what you are doing.__*

2) Clone the repo

```shell
git clone https://github.com/Rolv-Arild/rocket-learn.git
```

3) Start up, in order:

- The Redis server
- The Learner
- The Workers

Look at the examples to get up and running.
