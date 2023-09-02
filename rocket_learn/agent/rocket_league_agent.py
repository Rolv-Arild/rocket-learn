from typing import Set, Dict, Any

from rlgym.rocket_league.api import GameState

from rocket_learn.agent.agent import Agent


class RocketLeagueAgent(Agent):
    # Just to have more specific typing
    def reset(self, initial_state: GameState, agents: Set[str]):
        raise NotImplementedError

    # The "Any"s here will generally be (GameState, Scoreboard) and ndarray respectively but it's not forced
    def act(self, agents_observations: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def end(self, final_state: GameState, truncated: Dict[str, bool]):
        raise NotImplementedError
