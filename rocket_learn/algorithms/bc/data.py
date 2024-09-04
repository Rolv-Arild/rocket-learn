from pathlib import Path
from typing import List

import numpy as np
from rlgym.api import ObsBuilder, SharedInfoProvider
from rlgym_tools.replays.convert import replay_to_rlgym, get_valid_action_options
from rlgym_tools.replays.parsed_replay import ParsedReplay
from tqdm import tqdm

from rocket_learn.utils.util import conditional_stack


def make_bc_dataset(replay_paths: List[str], output_dir: str,
                    obs_builder: ObsBuilder, shared_info_provider: SharedInfoProvider,
                    action_options: np.ndarray,
                    label_smoothing: bool = False):
    output_dir = Path(output_dir)
    pbar = tqdm(replay_paths, "Processing replays")
    for replay_path in pbar:
        replay_path = Path(replay_path)
        parsed_replay = ParsedReplay.load(replay_path)
        iterator = replay_to_rlgym(parsed_replay)

        all_observations = {}
        all_actions = {}
        do_reset = True
        shared_info = shared_info_provider.create({})
        for replay_frame in iterator:
            game_state = replay_frame.state
            replay_actions = replay_frame.actions
            scoreboard = replay_frame.scoreboard
            agent_ids = list(game_state.cars.keys())
            if do_reset:
                shared_info_provider.set_state(agent_ids, game_state, shared_info)
                obs_builder.reset(agent_ids, game_state, shared_info)
                do_reset = False
                for agent_id in agent_ids:
                    all_observations[agent_id] = []
                    all_actions[agent_id] = []

            shared_info_provider.step(agent_ids, game_state, shared_info)
            shared_info["scoreboard"] = scoreboard  # Might overwrite, but this should be more accurate
            observations = obs_builder.build_obs(agent_ids, game_state, shared_info)

            for agent_id in agent_ids:
                all_observations[agent_id].append(observations[agent_id])

                replay_action = replay_actions[agent_id]
                car = game_state.cars[agent_id]
                mask, is_optimal = get_valid_action_options(car, replay_action, action_options)
                if label_smoothing:
                    action = mask.astype(np.float32)
                else:
                    distances = np.linalg.norm(action_options - replay_action, axis=1)
                    distances[~mask] = np.inf
                    best_actions = np.where(distances == distances.min())[0]
                    action = np.random.choice(best_actions)  # Random to not bias towards low indices

                all_actions[agent_id].append(action)

            if scoreboard.go_to_kickoff:
                do_reset = True

        for agent_id in all_observations.keys():
            np.savez_compressed(output_dir / f"{replay_path.stem}_{agent_id}.npz",
                                obs=conditional_stack(all_observations[agent_id]),
                                action=np.stack(all_actions[agent_id]))
