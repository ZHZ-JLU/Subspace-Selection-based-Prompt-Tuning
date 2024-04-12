import torch
import torch.nn as nn

class MLP_classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.output(x)
        return x