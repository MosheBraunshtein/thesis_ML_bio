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
            raise FileNotFoundError(f"\n Dataset directory not found: {self.dataset_path}\n")
        self.count_files = 0
        self.labels = []
        self.kinematics = []
        print(f"Extract {dataset_name} dataset ...\n")

    def extract(self, task, verbose = False, dtype = torch.float32):
        """Extracts data based on dataset and task."""
        print(f"Extract {task} task ... \n ")
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
            samples = []
            self.count_files += 1

            try:
                with open(file_path_transcription, 'r') as file_r:
                    for line in file_r:
                        start, end, label = line.split()
                        samples.append((int(start), int(end), label))
                        
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {file_path_transcription}")        

            # extract kinematics  
            try:      
                kinematics = np.loadtxt(file_path_kinematic)
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {file_path_kinematic}, transcriptions folder has this file but kinematic hasn't")

            
            if verbose : print(file + " " + file_r +" has " + self.count_files + " gestures")

            # cut kinematics by transcription
            for sample in samples:
                start, end, label = sample

                num_label = label_to_number(label)

                start_idx = start - 1
                end_idx = end
                
                rows = kinematics[start_idx:end_idx, :]    

                gestures_list.append(num_label)
                kinematics_list.append(rows)

        self.labels = torch.tensor(gestures_list,dtype=torch.int64)
        self.kinematics = kinematics_list
        print("gestures: ",set(gestures_list))

        assert len(self.labels) == len(self.kinematics), "target and data have different size"

        print("extract ", len(self.kinematics), " samples\n")
        return self.kinematics, self.labels

            

    def get_files_counts(self):
        return self.count_files
    
    def print_type_size(self):
        print(f"data type {type(self.kinematics)}, data size {len(self.kinematics)}")
        print(f"targets type {type(self.labels)},targets size ",self.labels.size()) 


    















