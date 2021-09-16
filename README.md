# rocket-learn
RLGym training tool

## TODO
- Add logging (✔ wandb only)
  - Give both PPO and RolloutGenerator access ✔
- Add reward normalization (and distribution?)
- Model freedom
  - Multiple inputs ✔ (when obs is tuple it batches them one by one)
  - Allow shared layers (ex.: `PPOAgent(shared, actor, critic)`) ✔
  - Recurrent?
  - Continuous actions if we really want
- Redis features 
  - Saving and loading (put everything we need into redis and call redis.save(), figure out loading)
  - Full setup (architecture, params, config?) communicated via redis, can start worker with only IP
  - Version quality is most important measurement, need to log it ✔
  - Implement quality update ✔
- Long-term plan is to set up a stream and let (at least some) people contribute with rollouts
  - Keep track of who is contributing ✔, make on-screen leaderboards
  - Exact setup should probably be in different repo
  - Rolv can keep it running on current PC, planning to get new one
  - Need to come to agreement on config (architecture, reward func, parameters etc.)
  - See `serious.py` for suggestions
  - Need nice name for the bot
- Known issues
  - Invalid values in logits appear very rarely, and seemingly at random
  - Policy update is super slow (5 mins+ using serious.py)
  - Qualities are not updated correctly in wandb
