from torch.utils.data import Dataset

class KinematicsDataset(Dataset):
    def __init__(self, X, y):
        self.X = X # torch tensor: [539, 2660, 76]
        self.y = y  # torch tensor: [539]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]