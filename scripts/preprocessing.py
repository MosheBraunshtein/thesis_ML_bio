import numpy as np
from scripts.config import DATA_PATH, label_to_number
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from scipy.interpolate import PchipInterpolator


class PreProcessing:
    """ this class provides preprocessing tools """

    def __init__(self,dataset) -> None:
        self.data = dataset

    
    def to_tensor(self,dtype = torch.float32):
        print("convert dataset to tensor type float32 ...\n")
        return [torch.tensor(arr,dtype=dtype) for arr in self.data]

    def padding_dataset(self,padded_value=0):
        """ padding all sequences with value in order to ger fix sequences length"""
        tensor_list = self.to_tensor()
        print("padding dataset with value ",padded_value," ...\n")
        self.data = pad_sequence(tensor_list, batch_first=True, padding_value=0)
        


    def pchip(self):
        """ Piecewise Cubic Hermite Interpolating Polynomial interpolation """
        print("PCHIP processing ...\n")

        # the logical sense is to create more time steps 
        max_timestep_sample = max(self.data, key=lambda x: x.shape[0])
        max_timestep = max_timestep_sample.shape[0]
        interpolate_data = []
        for sample in self.data:
            len = sample.shape[0]
            t_original = np.linspace(0, 1, len)  # Original time steps
            t_target = np.linspace(0, 1, max_timestep)        # New time steps
            
            interpolated_features = np.zeros((max_timestep,76))
            for feature in range(76):
                pchip_interpolator = PchipInterpolator(t_original,sample[:,feature])
                interpolated_features[:,feature] = pchip_interpolator(t_target)

            interpolate_data.append(interpolated_features)

        self.data = interpolate_data
        self.data = self.to_tensor()
        

                 
    def normalization(self):
        """ normalize dataset """
        print("normalize dataset ...")

    def print_data_type_and_size(self):
        print("data type ",type(self.data))
        print("data element type ",type(self.data[0]))    
