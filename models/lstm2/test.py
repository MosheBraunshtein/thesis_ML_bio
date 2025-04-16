import torch
import torch.nn as nn
from models.lstm2.model import LSTMModel
from torch.utils.data import DataLoader
from scripts.data_extractor import DataExtractor
from scripts.preprocessing import PreProcessing
from torch.utils.data import random_split
from scripts.dataset_creator import KinematicsDataset
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scripts.metrics import ModelEvaluator



# extract data    
extractor = DataExtractor("JIGSAWS")
data,targets = extractor.extract(task="suturing") 

# preprocessing 
preprocess = PreProcessing(data)
preprocess.pchip()
preprocess.normalization()
# preprocess.print_features_mean_std_over_a_sample()

preprocess.to_tensor()

# create dataset object
dataset = KinematicsDataset(preprocess.data, targets)

# split to train_dataset , test_dataset 
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size
print(f"#_train_dataset : {train_size}\n#_test_dataset : {test_size}\n")

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

assert test_size == test_loader.batch_size, "the evaluator output not include all test data"


# load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel().to(device)

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train
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

    acc = correct / train_size
    loss_value = running_loss/len(train_loader)
    print(f"Epoch {epoch+1}: Loss = {loss_value:.4f}, Accuracy = {acc:.4f}")


# save model parameters
save_path = os.path.join(os.path.dirname(__file__), "saved_model_parameters", "model.pth")
torch.save(model.state_dict(), save_path)
print("\nmodel parameters saved in: ")
print(os.path.abspath("model.pth"))

# test 
model.eval() 
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

# metrics
evaluator = ModelEvaluator(labels,preds)

evaluator.evaluate()
