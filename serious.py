# Preliminary setup for serious crowd-sourced model
# Exact setup should probably be in different repo
from earl_pytorch import EARLPerceiver, ActionOutputDiscrete
from torch.nn import Sequential
from rlgym.utils import ObsBuilder, TerminalCondition, RewardFunction, StateSetter

class SeriousObsBuilder(ObsBuilder):
  def reset():
    # Base on existing EARLObsBuilder
    # With EARLPerceiver we can use relative coords+vel(+more?) for key/value tensor, might be smart
    raise NotImplementedError

class SeriousTerminalCondition(TerminalCondition):  # What a name
  def reset():
    # Probably just use simple goal and no touch terminals
    raise NotImplementedError

class SeriousRewardFunction(RewardFunction):
  def reset():
    # Something like DistributeRewards(EventReward(goal=4, shot=4, save=4, demo=4, touch=1)) but find a way to reduce dribble abuse
    # Also record std/max/min rewards so we can actually see progress
    raise NotImplementedError

class SeriousStateSetter(StateSetter):
  def reset():
    # Use anything other than DefaultState?
    raise NotImplementedError

if __name__ == "__main__":  
  wandb.login(key=os.environ["WANDB_KEY"])
  logger = wandb.init(project="rocket-learn", entity="rolv-arild")

  rollout_gen = RedisRolloutGenerator(password="rocket-learn", logger=logger, save_every=1)

  agent = PPOAgent(
    actor=Sequential(Dense(256, 256), ActionOutputDiscrete(256)), 
    critic=Sequential(Dense(256, 256), Dense(256, 1)), 
    shared=EARLPerceiver(256)
  )
  alg = PPO(rollout_gen, agent, n_steps=1_000_000, batch_size=10_000, lr_critic=1e-5, lr_actor=1e-5, epochs=10, logger=logger)

  log_dir = "E:\\log_directory\\"
  repo_dir = "E:\\repo_directory\\"

  alg.run()

