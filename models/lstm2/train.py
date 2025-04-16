import torch
import torch.nn as nn
from models.lstm2.model import LSTMModel
from torch.utils.data import Dataset, DataLoader
from scripts.data_extractor import DataExtractor
from scripts.preprocessing import PreProcessing
from scripts.dataset_creator import KinematicsDataset
import os


# extract data    
extractor = DataExtractor("JIGSAWS")
data,targets = extractor.extract(task="suturing") 

# preprocessing 
preprocess = PreProcessing(data)
preprocess.pchip()
preprocess.normalization()

# preprocess.print_features_mean_std_over_a_sample()
preprocess.to_tensor()

dataset = KinematicsDataset(preprocess.data, targets) # preprocess.data = torch tensor: [#_samples, sequence length, #_input_size_features]
print(f"#_train_dataset : {len(dataset)}\n")
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel().to(device) 

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 15

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs) # tensor(batch_size,probability_value_for_each_class=12)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()

    acc = correct / len(dataset)
    print(f"Epoch {epoch+1}: Loss = {running_loss/len(train_loader):.4f}, Accuracy = {acc:.4f}")

# save model parameters
save_path = os.path.join(os.path.dirname(__file__), "saved_model_parameters", "model.pth")
torch.save(model.state_dict(), save_path)
print("\nmodel parameters saved in: ")
print(os.path.abspath("model.pth"))