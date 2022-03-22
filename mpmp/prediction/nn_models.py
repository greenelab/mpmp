import torch.nn as nn
import torch.nn.functional as F

class LogisticRegression(nn.Module):
    """Model for PyTorch logistic regression."""

    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        # one output for binary classification
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


class ThreeLayerNet(nn.Module):
    """Three-layer MLP for classification."""

    def __init__(self, input_size, h1_size=None, dropout=0.5):
        super(ThreeLayerNet, self).__init__()
        # three layers of decreasing size
        if h1_size is None:
            h1_size = input_size // 2
        self.fc0 = nn.Linear(input_size, h1_size)
        self.fc1 = nn.Linear(h1_size, h1_size // 2)
        self.fc2 = nn.Linear(h1_size // 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

