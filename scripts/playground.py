from scripts.data_extractor import DataExtractor
import torch
from models.lstm2.model import LSTMModel 


# extractor = DataExtractor("JIGSAWS")
# kinematics,gestures = extractor.extract(task="Knot_Tying")

# print("kinematic shape: ",kinematics.shape)
# print("gestures shape: ",gestures.shape)

model = LSTMModel()
model.print_parameters_data_type()
# for name, param in model.named_parameters():
#     print(f"Parameter: {name}, Data Type: {param.dtype}")


