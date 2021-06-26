class ExperienceBuffer:
    def __init__(self, meta):
        self.meta = meta
        self.result = 0
        self.observations = []
        self.actions = []
        self.rewards = []

    def add_step(self, observation, action, reward):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
