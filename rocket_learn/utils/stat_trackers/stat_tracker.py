import abc

import numpy as np


class StatTracker(abc.ABC):
    def __init__(self, name):
        self.name = name

    def reset(self):  # Called whenever
        raise NotImplementedError

    def update(self, gamestates: np.ndarray, masks: np.ndarray):
        raise NotImplementedError

    def get_stat(self):
        raise NotImplementedError
