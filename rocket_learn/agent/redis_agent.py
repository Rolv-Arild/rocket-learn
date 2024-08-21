from rocket_learn.agent.torch_agent import TorchPPOAgent


class RedisAgent(TorchPPOAgent):
    def __init__(self, model, obs_builder, action_parser, reward_function, redis_client, redis_key):
        super().__init__(model, obs_builder, action_parser, reward_function)
        self.redis_client = redis_client
        self.redis_key = redis_key

    def infer(self, obs, shared_info):
        obs = {k: obs[k].tolist() for k in obs}
        self.redis_client.set(self.redis_key, obs)
        return self.redis_client.get(self.redis_key)
