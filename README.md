# rocket-learn
RLGym training tool

## TODO
- Add logging (tensorboard only is fine initially)
- Add reward normalization (and distribution?)
- Model freedom
  - Allow shared layers (ex.: `PPOAgent(shared, actor, critic)`)
  - Continuous actions if we really want
- Redis features 
  - Long-term plan is to set up a stream and let (at least some) people contribute with rollouts
    - Need to come to agreement on config (architecture, reward func, parameters etc.)
    - Keep track of who is contributing, make on-screen leaderboards
    - Exact setup should probably be in different repo
    - Rolv can keep it running on old PC once he gets new one
  - Full setup (architecture, params, config?) communicated via redis, can start worker with only IP
  - Version quality is most important measurement, need to log it
