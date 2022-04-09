rocket-learn expects actor networks to output a set of probablities for each possible action. For example,
the default Discrete Action allows 8 actions, 5 of which are discrete control choices and 3 
of which are boolean choices. Because the Discrete control choices can each be -1, 0, or 1 and each 
boolean can be True or False, the network must output ((5 * 3) + (3 * 2)) aka 21 total actions. The actions 
must then be split into properly sized groups for each actions.

split = (3, 3, 3, 3, 3, 2, 2, 2)
total_output = sum(split)

class SplitLayer(nn.Module):
    def __init__(self, splits=(3, 3, 3, 3, 3, 2, 2, 2)):
        super().__init__()
        self.splits = splits

    def forward(self, x):
        return torch.split(x, self.splits, dim=-1)

actor = DiscretePolicy(nn.Sequential(
    nn.Linear(INPUT_SIZE, 256),
    nn.ReLU(),
    nn.Linear(256, total_output),
    SplitLayer(split)
), split)

As another example, KBM actions allow 2 Discrete controls and 3 boolean controls so the network must
output ((2 * 3) + (3 * 2)) aka 12 total actions

split = (3, 3, 2, 2, 2)
total_output = sum(split)

class SplitLayer(nn.Module):
    def __init__(self, splits=(3, 3, 3, 3, 3, 2, 2, 2)):
        super().__init__()
        self.splits = splits

    def forward(self, x):
        return torch.split(x, self.splits, dim=-1)

actor = DiscretePolicy(nn.Sequential(
    nn.Linear(INPUT_SIZE, 256),
    nn.ReLU(),
    nn.Linear(256, total_output),
    SplitLayer(split)
), split)