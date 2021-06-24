import os

from rlgym.envs import Match
from rlgym.gym import Gym

from MPFramework import MPFProcess


class Worker(MPFProcess):
    def __init__(self, epic_rl_path, **match_args):
        super().__init__("rocket-learn-worker")
        self.env = Gym(match=Match(**match_args), pipe_id=os.getpid(), path_to_rl=epic_rl_path, use_injector=True)
        self.agents = None
        self.rollouts = None

    def init(self):
        self.task_checker.wait_for_initialization("worker_init")

        self._data = self.task_checker.latest_data

    def update(self, header, data):
        if header == "agents":
            self._data = data

    def step(self):
        self.rollouts = self._collect_episode()

    def publish(self):
        if self.rollouts is not None:
            self.results_publisher.publish(self.rollouts, header="rollout_header")
            self.rollouts = None

    def cleanup(self):
        pass

    def _collect_episode(self):
        observations = self.env.reset()
        done = False
        rollouts = [[] for _ in self.agents]
        while not done:
            actions = [agent.get_action(agent.get_action_distribution(obs))
                       for agent, obs in zip(self.agents, observations)]
            observations, rewards, done, info = self.env.step(actions)
            for rollout, obs, act, rew in zip(rollouts, observations, rewards):
                rollout.append((obs, act, rew))
        return rollouts
