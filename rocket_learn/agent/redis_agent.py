from abc import ABC
from typing import Any, Dict

from redis import Redis

from rocket_learn.agent.policy import Policy
from rocket_learn.agent.torch_agent import TorchAgent
from rocket_learn.utils.experience_buffer import ExperienceBuffer
from rocket_learn.rollout_generator.redis.utils import encode_buffers, _serialize, ROLLOUTS


def send_experience_buffers(redis: Redis, identifiers: list[str], experience_buffers: list[ExperienceBuffer],
                            send_obs: bool, send_states: bool):
    rollout_data = encode_buffers(experience_buffers,
                                  return_obs=send_obs,
                                  return_states=send_states,
                                  return_rewards=True)
    # sanity_check = decode_buffers(rollout_data, versions,
    #                               has_obs=False, has_states=True, has_rewards=True,
    #                               obs_build_factory=lambda: self.match._obs_builder,
    #                               rew_func_factory=lambda: self.match._reward_fn,
    #                               act_parse_factory=lambda: self.match._action_parser)
    rollout_bytes = _serialize((rollout_data, identifiers,
                                send_obs, send_states, True))

    # TODO async communication?

    n_items = redis.rpush(ROLLOUTS, rollout_bytes)
    if n_items >= 1000:
        print("Had to limit rollouts. Learner may have have crashed, or is overloaded")
        redis.ltrim(ROLLOUTS, -100, -1)


class RedisAgent(TorchAgent, ABC):
    def __init__(self, policy: Policy,
                 redis: Redis, send_obs: bool = True, send_states: bool = True):
        super().__init__(policy)
        self.redis = redis

        self.send_obs = send_obs
        self.send_states = send_states

    def end(self, final_state: Any, truncated: Dict[str, bool]):
        if self.redis is not None:
            buffers = list(self._experience_buffers.values())
            identifiers = [self.identifier] * len(buffers)  # TODO get identifier here somehow
            send_experience_buffers(self.redis, identifiers, buffers, self.send_obs, self.send_states)
