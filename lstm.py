import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


data = np.sin(np.linspace(0, 100, 1000))  # Sine wave data
sequence_length = 10

# Prepare the data for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(data, sequence_length)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # Shape: (batch_size, seq_length, input_size)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)   # Shape: (batch_size, output_size)

print(X.shape)
print(y.shape)
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # last time step's output
        return out

# Instantiate the model
model = LSTMModel()


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 200

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')



model.eval()
with torch.no_grad():
    predictions = model(X)

# Convert predictions to numpy for visualization
predictions = predictions.numpy()

import matplotlib.pyplot as plt

plt.plot(data[sequence_length:], label='Actual Data')
plt.plot(predictions, label='Predicted Data')
plt.legend()
plt.show()
