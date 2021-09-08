from typing import Any

import cloudpickle


class CloudpickleWrapper:
    """
    ** Copied from SB3 **
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    :param var: the variable you wish to wrap for pickling with cloudpickle
    """

    def __init__(self, var: Any):
        self.var = var

    def __getstate__(self) -> Any:
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var: Any) -> None:
        self.var = cloudpickle.loads(var)