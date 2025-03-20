import torch
from scripts.data_extractor import DataExtractor
from models.lstm2.model import LSTMModel



extractor = DataExtractor("JIGSAWS")
data,targets = extractor.extract(task="needle_passing",batch_size=1)

model = LSTMModel().to(dtype=torch.float32)

input = data[0].unsqueeze(0)  # (batch_size=1,sequential_length=2660,num_features=76)

output = model(x=input)

print(output)

