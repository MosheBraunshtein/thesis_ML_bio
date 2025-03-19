import numpy as np
import os
from pathlib import Path
from scripts.config import DATA_PATH


# input: dataset name, task name 
# output: dataset

# Example: 
#   input: "JIGSAWS" , "Knot_Tying"
#   output: 
#           dictionary
#           { "G11" : [[],[],[]]
#             "G12" : [[],[],[]]
#           }


project_root = Path(__file__).resolve().parent.parent
base_path = os.path.join("datasets", "JIGSAWS", "Knot_Tying", "transcriptions")
file_path_gestures = os.path.join(base_path, "Knot_Tying_B001.txt")
print(file_path_gestures)
exit(0)
labeling = []
with open(file_path_gestures, 'r') as file:
    for line in file:
        start, end, classification = line.split()
        labeling.append((int(start), int(end), classification))
print(labeling)
exit(0)
# Specify the folder path 
folder_path = r'C:\Users\win10\Desktop\BGU_MSE\thesis_ML_bio\datasets\JIGSAWS\Knot_Tying\kinematics'
files = os.listdir(folder_path)

for file in files:
    file_path_kinematics = r'C:\Users\win10\Desktop\BGU_MSE\thesis_ML_bio\datasets\JIGSAWS\Knot_Tying\kinematics' + '\\' + file
    kinematics = np.loadtxt(file_path_kinematics)
    print("Data shape:", kinematics.shape)

    file_path_gestures =  r'C:\Users\win10\Desktop\BGU_MSE\thesis_ML_bio\datasets\JIGSAWS\Knot_Tying\transcriptions' + '\\' + file
    labeling = []
    with open(file_path_gestures, 'r') as file:
        for line in file:
            start, end, classification = line.split()
            labeling.append((int(start), int(end), classification))
    print(labeling)
    break
exit(0)

print("Data shape:", data.shape)

file = r'\Knot_Tying_B001.txt'
# load recorded kinematics
file_path_kinematics = r'C:\Users\win10\Desktop\BGU_MSE\thesis_ML_bio\datasets\JIGSAWS\Knot_Tying\Knot_Tying\kinematics\AllGestures' + file
data = np.loadtxt(file_path_kinematics)
print("Data shape:", data.shape)

# Load the classification file
labeling = []
file_path_classification = r'C:\Users\win10\Desktop\BGU_MSE\thesis_ML_bio\datasets\JIGSAWS\Knot_Tying\Knot_Tying\transcriptions\Knot_Tying_B001.txt'
with open(file_path_classification, 'r') as file:
    for line in file:
        start, end, classification = line.split()
        labeling.append((int(start), int(end), classification))

# Create a dictionary to hold classified data
classified = {}

for start, end, label in labeling:
    # Adjust for 1-based to 0-based indexing
    start_idx = start - 1
    end_idx = end
    
    # Extract rows
    rows = data[start_idx:end_idx, :]
    
    # Store in the dictionary

    if label not in classified:
        classified[label] = []
    
    classified[label].append(rows)


print(len(classified['G12']))
