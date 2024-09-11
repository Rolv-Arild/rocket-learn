import glob
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional

import numpy as np
from rlgym.api import ObsBuilder, SharedInfoProvider
from rlgym_tools.action_parsers.advanced_lookup_table_action import AdvancedLookupTableAction
from rlgym_tools.obs_builders.relative_default_obs import RelativeDefaultObs
from rlgym_tools.replays.convert import replay_to_rlgym, get_valid_action_options
from rlgym_tools.replays.parsed_replay import ParsedReplay
from tqdm import tqdm

from rocket_learn.utils.util import conditional_stack


def process_single_replay(replay_path: str, output_dir: str,
                          obs_builder: ObsBuilder,
                          shared_info_provider: Optional[SharedInfoProvider],
                          action_options: np.ndarray,
                          label_smoothing: bool = False,
                          overwrite: bool = False):
    replay_path = Path(replay_path)
    output_dir = Path(output_dir)
    replay_id = replay_path.stem
    out_path = output_dir / f"{replay_id}.npz"
    if not overwrite and out_path.exists():
        return
    parsed_replay = ParsedReplay.load(replay_path)
    iterator = replay_to_rlgym(parsed_replay)

    all_observations = {}
    all_actions = {}
    episode_ids = []
    steps = []
    step = 0
    episode_id = 0
    shared_info = shared_info_provider.create({}) if shared_info_provider is not None else {}
    for replay_frame in iterator:
        game_state = replay_frame.state
        replay_actions = replay_frame.actions
        scoreboard = replay_frame.scoreboard
        agent_ids = list(game_state.cars.keys())
        if step == 0:
            if shared_info_provider is not None:
                shared_info_provider.set_state(agent_ids, game_state, shared_info)
            obs_builder.reset(agent_ids, game_state, shared_info)
            for agent_id in agent_ids:
                all_observations[agent_id] = []
                all_actions[agent_id] = []

        shared_info["scoreboard"] = scoreboard
        if shared_info_provider is not None:
            shared_info_provider.step(agent_ids, game_state, shared_info)
        observations = obs_builder.build_obs(agent_ids, game_state, shared_info)

        for agent_id in agent_ids:
            all_observations[agent_id].append(observations[agent_id])

            replay_action = replay_actions[agent_id]
            car = game_state.cars[agent_id]
            mask, is_optimal = get_valid_action_options(car, replay_action, action_options)
            if label_smoothing:
                action = mask
            else:
                # Use L2 distance on valid actions to find the closest one
                distances = np.linalg.norm(action_options - replay_action, axis=1)
                distances[~mask] = np.inf
                best_actions = np.where(distances - distances.min() < 1e-3)[0]
                action = np.random.choice(best_actions)  # Random to not bias towards low indices
                action = int(action)

            all_actions[agent_id].append(action)

        episode_ids.append(episode_id)
        steps.append(step)

        if scoreboard.go_to_kickoff:
            step = 0
            episode_id += 1
        else:
            step += 1

    np.savez_compressed(
        out_path,
        observations={k: conditional_stack(v) for k, v in all_observations.items()},
        actions={k: conditional_stack(v) for k, v in all_actions.items()},
        episode_ids=episode_ids,
        steps=steps,
        replay_id=replay_id
    )


def make_bc_dataset(replay_paths: List[str], output_dir: str,
                    obs_builder: ObsBuilder,
                    shared_info_provider: Optional[SharedInfoProvider],
                    action_options: np.ndarray,
                    label_smoothing: bool = False,
                    overwrite: bool = False,
                    n_proc: int = None):
    # pbar = tqdm(replay_paths, "Processing replays")
    # for replay_path in pbar:
    #     process_single_replay(replay_path, output_dir, obs_builder, shared_info_provider, action_options,
    #                           label_smoothing, overwrite)
    with ProcessPoolExecutor(n_proc) as ex:
        futures = []
        for replay_path in replay_paths:
            futures.append(ex.submit(process_single_replay, replay_path, output_dir,
                                     obs_builder, shared_info_provider, action_options, label_smoothing, overwrite))
        for future in futures:
            future.result()
            pbar.update()


def main():
    replay_files = glob.glob(r"E:\rokutleg\replays\RLCS\RLCS 2024\**\*.replay", recursive=True)
    output_dir = r"E:\rokutleg\datasets\bc\RLCS 2024"
    os.makedirs(output_dir, exist_ok=True)
    obs_builder = RelativeDefaultObs()
    shared_info_provider = None
    action_options = AdvancedLookupTableAction.make_lookup_table(include_stalls=True)
    make_bc_dataset(replay_files, output_dir, obs_builder, shared_info_provider, action_options,
                    label_smoothing=False, overwrite=True)


if __name__ == '__main__':
    main()
