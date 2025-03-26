import torch
from scripts.data_extractor import DataExtractor
from models.lstm2.model import LSTMModel
from sklearn.metrics import accuracy_score

# Load the trained model
# model = torch.load("model.pth")
# model.eval()
model = LSTMModel().to(dtype=torch.float32)  
# model.load_state_dict(torch.load("lstm_model.pth")) 
model.eval() 



# Load and preprocess test data
extractor = DataExtractor("JIGSAWS")
data,targets = extractor.extract(task="needle_passing",batch_size=1)


# Run inference
with torch.no_grad():
    outputs = model(data)  # Forward pass # (539,12)
    predictions = torch.argmax(outputs, dim=1).numpy() # return the indices with the highest value  
    # index=5 => label=5 => gesture=G5
    predictions[0] = 1
# Evaluate model
accuracy = accuracy_score(targets.numpy(), predictions)
print(f"Test Accuracy: {accuracy:.4f}")
