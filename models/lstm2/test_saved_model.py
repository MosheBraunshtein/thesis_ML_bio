import torch
from models.lstm2.model import LSTMModel
import os
from scripts.data_extractor import DataExtractor
from scripts.preprocessing import PreProcessing
import numpy as np
from scripts.dataset_creator import KinematicsDataset
from torch.utils.data import DataLoader
from scripts.metrics import ModelEvaluator

# load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel().to(device)

lstm2_dir = os.path.dirname(__file__)
saved_parameters_path = os.path.join(lstm2_dir,"saved_model_parameters","model.pth") 
model.load_state_dict(torch.load(saved_parameters_path))


# load dataset to test

## extract data    
extractor = DataExtractor("JIGSAWS")
data,targets = extractor.extract(task="Needle_Passing") 

## preprocessing 
preprocess = PreProcessing(data)
preprocess.pchip()
preprocess.normalization()
# preprocess.print_features_mean_std_over_a_sample()
preprocess.to_tensor()

## create dataset object
test_dataset = KinematicsDataset(preprocess.data, targets)
test_size = len(test_dataset)
print(f"#_train_dataset : {test_size}\n")
test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

assert test_size == test_loader.batch_size, "the evaluator output not include all test data"

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


