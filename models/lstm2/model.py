import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scripts.config import KINEMATICS_FEATURES_SIZE, GESTURES_SIZE


class LSTMModel(nn.Module):
    def __init__(self, input_size=KINEMATICS_FEATURES_SIZE, hidden_size=50, output_size=GESTURES_SIZE, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x, h0, c0):
        out, _ = self.lstm(x, (h0, c0)) # out = contains the hidden state output for each time step in the input sequence.
        last_step_feature_output = out[:, -1, :] # selects the hidden state output from the last time step of the sequence for each batch
        out = self.fc(last_step_feature_output) 
        return out
    
