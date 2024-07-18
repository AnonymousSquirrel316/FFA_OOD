
from abc import abstractmethod

import torch
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Normalize, Lambda

def _ttemp_flatten(x):
    return torch.flatten(x)

def _base_normalization(x):
    x -= torch.min(x, dim=1, keepdim=True)[0]
    x /= torch.max(x, dim=1, keepdim=True)[0]+0.001
    return x

class DataLoaderExtractor:
    
    transform = Compose([
                ToTensor(),
                Normalize((0.1307,), (0.3081,)),
                Lambda(_ttemp_flatten)])
    
    def __init__(self, batch_size = 64):
        self.batch_size = batch_size
        self.dataloader = None
        
    @abstractmethod
    def load_dataloader(self, download = False, split = None, **kwargs):
        pass
    
    def get_dataloader(self, **kwargs) -> DataLoader:
        self.load_dataloader(**kwargs)
        
        return self.dataloader