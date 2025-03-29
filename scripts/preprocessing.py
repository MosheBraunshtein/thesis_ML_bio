import numpy as np
from scripts.config import DATA_PATH, label_to_number
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from scipy.interpolate import PchipInterpolator
from scripts.config import KINEMATICS_FEATURES_SIZE
import matplotlib.pyplot as plt



class PreProcessing:
    """ this class provides preprocessing tools """

    def __init__(self,dataset) -> None:
        self.data = dataset
        self.data_distribution = []
        self.samples_size = len(dataset)
        max_timestep_sample = max(self.data, key=lambda x: x.shape[0])
        self.max_timestep = max_timestep_sample.shape[0]
        self.data_distribution_params = np.zeros((self.samples_size,KINEMATICS_FEATURES_SIZE,2))


    
    def to_tensor(self,dtype = torch.float32):
        print("convert dataset to tensor type float32 ...\n")
        self.data = [torch.tensor(arr,dtype=dtype) for arr in self.data]

    def padding_dataset(self,padded_value=0):
        """ padding all sequences with value in order to ger fix sequences length"""
        tensor_list = self.to_tensor()
        print("padding dataset with value ",padded_value," ...\n")
        self.data = pad_sequence(tensor_list, batch_first=True, padding_value=0)
        


    def pchip(self):
        """ Piecewise Cubic Hermite Interpolating Polynomial interpolation """
        print("PCHIP processing ...\n")

        # the logical sense is to create more time steps
        interpolate_data = []
        for sample in self.data:
            len = sample.shape[0]
            t_original = np.linspace(0, 1, len)  # Original time steps
            t_target = np.linspace(0, 1, self.max_timestep)        # New time steps
            
            interpolated_features = np.zeros((self.max_timestep,76))
            for feature in range(76):
                pchip_interpolator = PchipInterpolator(t_original,sample[:,feature])
                interpolated_features[:,feature] = pchip_interpolator(t_target)

            interpolate_data.append(interpolated_features)

        self.data = interpolate_data
        

                 
    def normalization(self):
        """ normalize dataset """
        print("normalize dataset ...")

        # calculate mean and deviation for each feature across a sample for all samples
        sample_i=0
        for sample in self.data : 
            for feature in range(KINEMATICS_FEATURES_SIZE):
                feature_over_time = sample[:,feature]
                feature_mean = np.mean(feature_over_time)
                feature_std = np.std(feature_over_time)
                assert feature_std != 0, "feature std is zero , can't normalize feature "
                # save distribution parameters
                self.data_distribution_params[sample_i,feature,:] = [feature_mean,feature_std]
                # normalize data
                self.data[sample_i][:,feature] =  (self.data[sample_i][:,feature] - feature_mean)/feature_std 
            sample_i += 1
        



    def print_data_type_and_size(self):
        print("data type ",type(self.data))
        print("data element type ",type(self.data[0]))    

    def print_features_mean_std_over_a_sample(self,sample=0):
        """choose sample to plot his features mean and std"""
        mean_values =  self.data_distribution_params[sample,:,0]
        std_values =  self.data_distribution_params[sample,:,1]

        plt.plot(mean_values, marker='o', linestyle='-')
        plt.xlabel("features")
        plt.ylabel(f"mean")
        plt.grid(True)
        plt.show()

        plt.plot(std_values, marker='o', linestyle='-')
        plt.xlabel("features")
        plt.ylabel(f"std")
        plt.grid(True)
        plt.show()
