from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(310, 1860),
            nn.ReLU(),
            nn.Linear(1860, 620),
            nn.ReLU(),
            nn.Linear(620, 105),
            nn.ReLU(),
            nn.Linear(105, 30),
            nn.ReLU(),
            nn.Linear(30, 9),
            nn.ReLU(),
            nn.Linear(9, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x.float())
        return logits
        