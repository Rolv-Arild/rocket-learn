class ExperienceBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    def discount_rewards(self, gamma):
        return []