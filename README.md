# rocket-learn

## What is rocket-learn?

rocket-learn is a machine learning framework specifically designed for Rocket League Reinforcement Learning. 
It works in conjunction with Rocket League, RLGym, and Bakkesmod.

## What features does rocket-learn have?

<ul>
<li>Reinforcement learning algorithm available out of the box</li>
  <ul>
    <li>Proximal Policy Optimization (PPO)</li>
    <li>extensible format allows new algorithms to be added</li>
  </ul>
<li>Distributed compute from multiple computers</li>
<li>Automatic saving of and training against previous agent versions</li>
<li>Trueskill progress tracking</li>
<li>Training against Hardcoded/Pretrained Agents</li>
<li>Training against Humans</li>
<li>Saving and loading models</li>
<li>wandb logging</li>
</ul>


## Should I use rocket-learn?

You should use Stable Baselines3 (SB3) to make your bot at first. The hardest parts of building a 
machine learning bot are 

- understanding how to program
- understanding how machine learning works
- choosing good hyperparameters
- choosing good reward functions
- choosing an action parser
- making a statesetter that puts the bot in the best situations

SB3 is a great way to figure out those essential parts. Once you have all of those aspects down, rocket-learn
may be a good next step to a better machine learning bot. 

If you *don't* yet have these, rocket-learn will add a large amount of complexity for no added benefit. It's 
important to remember that high compute and a tough opponent are less important than good fundamentals of ML.

## How do I setup rocket-learn?

1) Get [Redis](https://docs.servicestack.net/install-redis-windows) running 

*__Improper Redis setup can leave your computer extremely vulnerable to Bad Guys. 
We are not responsible for your computer's safety. We assume you know what you are doing.__*

2) Clone the repo

```
git clone https://github.com/Rolv-Arild/rocket-learn.git
```

3) Look at the examples to get up and running
