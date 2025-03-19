import torch
from scripts.data_extractor import DataExtractor
from models.lstm2.model import LSTMModel



extractor = DataExtractor("JIGSAWS")
kinematics,gesture = extractor.extract(task="needle_passing",batch_size=1)


model = LSTMModel().to(dtype=torch.float32)

input = kinematics[0].unsqueeze(1) # (sequential_length=421,batch_size=1,num_features=76)

# initial hidden units
h0 = torch.randn(model.num_layers,1,model.hidden_size,dtype=torch.float32)

# initial cell state 
c0 = torch.randn(model.num_layers,1,model.hidden_size,dtype=torch.float32)

# Debugging: Check Shapes and Data Types (for nn.LSTM(first_batch=False by defualt))
print("input shape:", input.shape, "dtype:", input.dtype)  # (seq_len, batch_size, input_size)
print("h0 shape:", h0.shape, "dtype:", h0.dtype)  # (num_layers, batch_size, hidden_size)
print("c0 shape:", c0.shape, "dtype:", c0.dtype)  # (num_layers, batch_size, hidden_size)
exit(0)
output = model(x=input,h0=h0,c0=c0)

print(output)

