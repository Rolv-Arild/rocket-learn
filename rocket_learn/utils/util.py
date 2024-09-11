import numpy as np
import torch

try:
    import numba

    optional_njit = numba.njit
except ImportError:
    def optional_njit(func):
        return func


@optional_njit
def calculate_advantages(gamma, gae_lambda, rewards, values, terminations, truncations):  # TODO verify
    """
    Calculate the advantages using GAE.

    :param gamma: The discount factor.
    :param gae_lambda: The GAE lambda.
    :param rewards: The rewards.
    :param values: The value estimates.
    :param terminations: The termination flags.
    :param truncations: The truncation flags.
    :return: The advantages.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)

    size = len(rewards)
    last_gae_lam = 0
    for i in reversed(range(size)):
        if i == size - 1:
            next_non_terminal = 1.0 - terminations[i]
            next_values = values[-1]
        else:
            next_non_terminal = (1.0 - terminations[i]) * (1.0 - truncations[i])
            next_values = values[i + 1]
        delta = rewards[i] + gamma * next_values * next_non_terminal - values[i]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[i] = last_gae_lam
    return advantages


def conditional_stack(data):
    assert len(data) > 0, "No data to stack"
    d0 = data[0]
    if isinstance(d0, (int, float)):
        return np.array(data)
    elif isinstance(d0, np.ndarray):
        return np.stack(data)
    elif isinstance(d0, torch.Tensor):
        return torch.stack(data)
    elif isinstance(d0, dict):
        return {
            key: conditional_stack([d[key] for d in data])
            for key in d0
        }
    elif isinstance(d0, (list, tuple)):
        return type(d0)(
            conditional_stack(d)
            for d in zip(*data)
        )
    else:
        raise ValueError(f"Unsupported type: {type(d0)}")


def transform_batch(batch, fn):
    if isinstance(batch, tuple):
        return tuple(fn(t) for t in batch)
    elif isinstance(batch, list):
        return [fn(t) for t in batch]
    elif isinstance(batch, dict):
        return {k: fn(v) for k, v in batch.items()}
    else:
        return fn(batch)


def make_table(versions, ratings, blue, orange, pretrained_choice):
    from tabulate import tabulate

    version_info = []
    for v, r in zip(versions, ratings):
        if pretrained_choice is not None and v == 'na':  # print name but don't send it back
            version_info.append([str(type(pretrained_choice).__name__), "N/A"])
        elif v == 'na':
            version_info.append(['Human', "N/A"])
        else:
            if isinstance(v, int) and v < 0:
                v = f"Latest ({-v})"
            version_info.append([v, f"{r.mu:.2f}±{2 * r.sigma:.2f}"])

    blue_versions, blue_ratings = list(zip(*version_info[:blue]))
    orange_versions, orange_ratings = list(zip(*version_info[blue:]))

    if blue < orange:
        blue_versions += [""] * (orange - blue)
        blue_ratings += [""] * (orange - blue)
    elif orange < blue:
        orange_versions += [""] * (blue - orange)
        orange_ratings += [""] * (blue - orange)

    table_str = tabulate(list(zip(blue_versions, blue_ratings, orange_versions, orange_ratings)),
                         headers=["Blue", "rating", "Orange", "rating"], tablefmt="rounded_outline")

    return table_str
