from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6, 36),
            nn.ReLU(),
            nn.Linear(36, 648),
            nn.ReLU(),
            nn.Linear(648, 36),
            nn.ReLU(),
            nn.Linear(36, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x.float())
        return logits