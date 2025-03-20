import numpy as np
from scripts.config import DATA_PATH, label_to_number
import os
import torch
from torch.nn.utils.rnn import pad_sequence

class DataExtractor:
    """A class for extracting dataset files dynamically based on dataset name and task."""
    
    def __init__(self, dataset_name):
        """Initialize with the dataset name and verify existence."""
        self.dataset_path = DATA_PATH / dataset_name
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_path}")
        self.count_files = 0

    def extract(self, task, batch_size=1, verbose = False, dtype = torch.float32):
        """Extracts data based on dataset and task."""

        kinematics_list = []
        gestures_list = []

        transcriptions_dir_path = self.dataset_path / f"{task}" / "transcriptions" 

        if not transcriptions_dir_path.exists():
            raise FileNotFoundError(f"Transcription folder not found: {transcriptions_dir_path}")
        
        kinematics_dir_path = self.dataset_path / f"{task}" / "kinematics"

        if not kinematics_dir_path.exists():
            raise FileNotFoundError(f"Kinematic Folder not found: {kinematics_dir_path}")

        files = os.listdir(transcriptions_dir_path)

        for file in files:
            file_path_transcription = transcriptions_dir_path / file
            file_path_kinematic = kinematics_dir_path / file
            classification = []
            self.count_files += 1

            try:
                with open(file_path_transcription, 'r') as file_r:
                    for line in file_r:
                        start, end, label = line.split()
                        classification.append((int(start), int(end), label))
                        
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {file_path_transcription}")        

            # extract kinematics  
            try:      
                kinematics = np.loadtxt(file_path_kinematic)
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {file_path_kinematic}, transcriptions folder has this file but kinematic hasn't")

            
            if verbose : print(file," ",self.count_files," has ",len(self.__classification)," gestures")

            # cut kinematics by transcription
            for item in classification:
                start, end, label = item

                num_label = label_to_number(label)

                start_idx = start - 1
                end_idx = end
                # assert end_idx-start_idx < 2000, file
                rows = kinematics[start_idx:end_idx, :]
                kinematics_of_gesture = torch.tensor(rows,dtype=dtype)

                gestures_list.append(num_label)
                kinematics_list.append(kinematics_of_gesture)

            padded_tensors = pad_sequence(kinematics_list, batch_first=True, padding_value=0)
            target_tensor = torch.tensor(gestures_list,dtype=dtype)

        return padded_tensors, target_tensor

            

    def extract_kinematics(self,task):
        kinematics_dir_path = self.dataset_path / f"{task}" / "kinematics"

        if not kinematics_dir_path.exists():
            raise FileNotFoundError(f"Kinematic Folder not found: {kinematics_dir_path}")
        
        kinematics_files = os.listdir(kinematics_dir_path)

        for file in kinematics_files:
            kinematics = np.loadtxt(kinematics_dir_path/file)
            self.kinematic_count += 1


    def extract_transcription(self,task):
        transcriptions_dir_path = self.dataset_path / f"{task}" / "transcriptions" 

        if not transcriptions_dir_path.exists():
            raise FileNotFoundError(f"Transcription folder not found: {transcriptions_dir_path}")
        
        transcriptions_files = os.listdir(transcriptions_dir_path)

        for file in transcriptions_files:
            file_path_transcription = transcriptions_dir_path / file

            with open(file_path_transcription, 'r') as file:
                for line in file:
                    start, end, classification = line.split()
                    self.__labeling.append((int(start), int(end), classification))
                self.trans_count += 1


    def get_files_counts(self):
        return self.count_files


    















