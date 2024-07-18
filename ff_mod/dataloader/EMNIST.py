from torchvision.datasets import EMNIST
from ff_mod.dataloader.base import DataLoaderExtractor, _ttemp_flatten

import torch
import torchvision.transforms as transforms


class EMNIST_Dataloader(DataLoaderExtractor):
    
    def __init__(self, batch_size = 64, **kwargs):
        super().__init__(batch_size=batch_size)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(_ttemp_flatten)
        ])
    
    def load_dataloader(self, download = False, split = None, **kwargs):
        dataset = EMNIST(
            './data',
            transform=self.transform,
            split = "letters",
            train=(split == "train"),
            download=download
        )
        
        # Change from [1,26] to [0,25]
        dataset.targets -= 1
        
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )