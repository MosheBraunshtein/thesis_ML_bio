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
        print("PCHIP processing ...")

                 
    def normalization(self):
        """ normalize dataset """
        print("normalize dataset ...")

    def print_data_type_and_size(self):
        print("data type ",type(self.data))
        print("data size ",self.data.size())    
