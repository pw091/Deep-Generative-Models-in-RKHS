import pandas as pd
import torch
import torchvision.transforms.functional as F

class MNISTDataset(Dataset):
    '''proprietary class; expects filepath to MNIST csv in original format'''
    
    def __init__(self, csv_file: str):
        data = pd.read_csv(csv_file)
        self.X = data.drop('label', axis=1).values / 255. #normalize pixel values
        self.y = torch.tensor(data['label'].values)
        self.y_one_hot = F.one_hot(self.y, num_classes=10).float() #one-hot label encoding

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.FloatTensor(self.X[idx]), self.y_one_hot[idx]