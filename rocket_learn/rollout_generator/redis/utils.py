# Constants for consistent key lookup
import pickle
import zlib
from typing import List, Optional, Union, Dict

import numpy as np
from redis import Redis
from rlgym_sim.utils.gamestates import GameState
from trueskill import Rating

from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.utils.batched_obs_builder import BatchedObsBuilder
from rocket_learn.utils.gamestate_encoding import encode_gamestate
import msgpack
import msgpack_numpy as m

QUALITIES = "qualities-{}"
N_UPDATES = "num-updates"
# SAVE_FREQ = "save-freq"
# MODEL_FREQ = "model-freq"

MODEL_LATEST = "model-latest"
VERSION_LATEST = "model-version"

ROLLOUTS = "rollout"
OPPONENT_MODELS = "opponent-models"
WORKER_IDS = "worker-ids"
CONTRIBUTORS = "contributors"
LATEST_RATING_ID = "latest-rating-id"
EXPERIENCE_PER_MODE = "experience-per-mode"
_ALL = (
    N_UPDATES, MODEL_LATEST, VERSION_LATEST, ROLLOUTS, OPPONENT_MODELS,
    WORKER_IDS, CONTRIBUTORS, LATEST_RATING_ID, EXPERIENCE_PER_MODE)

m.patch()


# Helper methods for easier changing of byte conversion
def _serialize(obj):
    return zlib.compress(msgpack.packb(obj), level=0)


def _unserialize(obj):
    return msgpack.unpackb(zlib.decompress(obj))


def _serialize_model(mdl):
    device = next(mdl.parameters()).device  # Must be a better way right?
    mdl_bytes = pickle.dumps(mdl.cpu())
    mdl.to(device)
    return mdl_bytes


def _unserialize_model(buf):
    agent = pickle.loads(buf)
    return agent


def get_rating(gamemode: str, model_id: Optional[str], redis: Redis) -> Union[Rating, Dict[str, Rating]]:
    """
    Get the rating of a player.
    :param gamemode: The game mode to get the rating for.
    :param model_id: The id of the model.
    :param redis: The redis client.
    :return: The rating of the player.
    """
    quality_key = QUALITIES.format(gamemode)
    if model_id is None:  # Return all ratings
        return {
            k.decode("utf-8"): Rating(*_unserialize(v))
            for k, v in redis.hgetall(quality_key).items()
        }
    return Rating(*_unserialize(redis.hget(quality_key, model_id)))


def encode_buffers(buffers: List[ExperienceBuffer], return_obs=True, return_states=True, return_rewards=True):
    res = []

    if return_states:
        states = np.asarray([info["numpy_state"] for info in buffers[0].infos] if len(buffers) > 0 else [])
        res.append(states)

    if return_obs:
        observations = [buffer.observations for buffer in buffers]
        res.append(observations)

    if return_rewards:
        rewards = np.asarray([buffer.rewards for buffer in buffers])
        res.append(rewards)

    actions = np.asarray([buffer.actions for buffer in buffers])
    log_probs = np.asarray([buffer.log_probs for buffer in buffers])
    dones = np.asarray([buffer.dones for buffer in buffers])
    res.append(actions)
    res.append(log_probs)
    res.append(dones)

    return res


def decode_buffers(enc_buffers, versions, has_obs, has_states, has_rewards,
                   obs_build_factory=None, rew_func_factory=None, act_parse_factory=None):
    assert has_states or has_obs, "Must have at least one of obs or states"
    assert has_states or has_rewards, "Must have at least one of rewards or states"
    assert not has_obs or has_obs and has_rewards, "Must have both obs and rewards"  # TODO obs+no reward?

    i = 0
    if has_states:
        game_states = enc_buffers[i]
        if len(game_states) == 0:
            raise RuntimeError
        i += 1
    else:
        game_states = None
    if has_obs:
        obs = enc_buffers[i]
        i += 1
    else:
        obs = None
    if has_rewards:
        rewards = enc_buffers[i]
        i += 1
        # dones = np.zeros_like(rewards, dtype=bool)  # TODO: Support for dones?
        # if len(dones) > 0:
        #     dones[:, -1] = True
    else:
        rewards = None
        # dones = None
    actions = enc_buffers[i]
    i += 1
    log_probs = enc_buffers[i]
    i += 1
    dones = enc_buffers[i]
    i += 1

    if obs is None:
        # Reconstruct observations
        obs_builder = obs_build_factory()
        act_parser = act_parse_factory()
        if isinstance(obs_builder, BatchedObsBuilder):
            # TODO support states+no rewards
            assert game_states is not None and rewards is not None, "Must have both game states and rewards"
            obs = obs_builder.batched_build_obs(game_states[:-1])
            prev_actions = act_parser.parse_actions(actions.reshape((-1,) + actions.shape[2:]).copy(), None).reshape(
                actions.shape[:2] + (8,))
            prev_actions = np.concatenate((np.zeros((actions.shape[0], 1, 8)), prev_actions[:, :-1]), axis=1)
            obs_builder.add_actions(obs, prev_actions)
            buffers = [
                ExperienceBuffer(observations=[obs[i]], actions=actions[i], rewards=rewards[i], dones=dones[i],
                                 log_probs=log_probs[i])
                for i in range(len(obs))
            ]
            return buffers, game_states
        else:  # Slow reconstruction, but works for any ObsBuilder
            gs_arrays = game_states
            game_states = [GameState(gs.tolist()) for gs in game_states]
            rew_func = rew_func_factory()
            obs_builder.reset(game_states[0])
            rew_func.reset(game_states[0])
            buffers = [
                ExperienceBuffer(infos=[{"state": game_states[0]}])
                for _ in range(len(game_states[0].players))
            ]

            env_actions = [
                act_parser.parse_actions(actions[:, s, :].copy(), game_states[s])
                for s in range(actions.shape[1])
            ]

            obss = [obs_builder.build_obs(p, game_states[0], np.zeros(8))
                    for i, p in enumerate(game_states[0].players)]
            for s, gs in enumerate(game_states[1:]):
                assert len(gs.players) == len(versions)
                final = s == len(game_states) - 2
                old_obs = obss
                obss = []
                i = 0
                for version in versions:
                    if version == 'na':
                        continue  # don't want to rebuild or use prebuilt agents
                    player = gs.players[i]

                    # IF ONLY 1 buffer is returned, need a way to say to discard bad version

                    obs = obs_builder.build_obs(player, gs, env_actions[s][i])
                    if rewards is None:
                        if final:
                            rew = rew_func.get_final_reward(player, gs, env_actions[s][i])
                        else:
                            rew = rew_func.get_reward(player, gs, env_actions[s][i])
                    else:
                        rew = rewards[i][s]
                    buffers[i].add_step(old_obs[i], actions[i][s], rew, final, log_probs[i][s], {"state": gs})
                    obss.append(obs)
                i += 1

            return buffers, gs_arrays
    else:  # We have everything we need
        buffers = []
        for i in range(len(obs)):
            buffers.append(
                ExperienceBuffer(observations=obs[i],
                                 actions=actions[i],
                                 rewards=rewards[i],
                                 dones=dones[i],
                                 log_probs=log_probs[i])
            )
        return buffers, game_states
