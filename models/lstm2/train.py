import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from scripts.data_extractor import DataExtractor
from model import LSTMModel

# Load your test dataset (replace with actual data loading logic)
def load_test_data(batch_size):
    extractor = DataExtractor("JIGSAWS")
    kinematics,gesture = extractor.extract(task="needle_passing") # (batch_size,)
    
    # to tensor 
    
    # Assuming X_test is a tensor of shape (batch_size, sequence_length, input_size)
    # Example shape: (batch_size=10, seq_len=5, input_size=3)
    gesture = torch.tensor(["G1","G2","G3","G4","G5","G6","G7","G8","G9","G10","G11","G12","G13","G14","G15"])
    return kinematics, gesture

def batch_data_set():
# Load trained model (replace 'model.pth' with your trained model file)
input_size = 3
hidden_size = 16
output_size = 2  # Adjust as needed
model = LSTMModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("model.pth"))
model.eval()  # Set to evaluation mode

# Load test data
X_test, y_test = load_test_data()

# Perform inference
with torch.no_grad():
    outputs = model(X_test)
    predicted = torch.argmax(outputs, dim=1)

# Print results
print("Predicted labels:", predicted.numpy())
print("Actual labels:", y_test.numpy())

# Optional: Compute accuracy
accuracy = (predicted == y_test).float().mean().item()
print(f"Test Accuracy: {accuracy * 100:.2f}%")
