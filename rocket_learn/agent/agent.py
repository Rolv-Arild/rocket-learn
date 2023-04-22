from typing import Any, Dict, Set


class Agent:
    def reset(self, initial_state: Any, agents: Set[str]):
        raise NotImplementedError

    def act(self, agents_observations: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def end(self, final_state: Any, truncated: Dict[str, bool]):
        raise NotImplementedError
