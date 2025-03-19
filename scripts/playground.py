from scripts.data_extractor import DataExtractor
import torch


extractor = DataExtractor("JIGSAWS")
kinematics,gestures = extractor.extract(task="needle_passing")

# print(gestures[200])