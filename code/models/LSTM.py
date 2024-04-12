import torch
import torch.nn as nn

class LSTM_classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_classifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = torch.cat((out[:, -1, :self.lstm.hidden_size], out[:, 0, self.lstm.hidden_size:]), dim=1)
        out = self.fc(out)
        x = self.output(out)
        return x