from torchvision.datasets import FashionMNIST

from ff_mod.dataloader.base import DataLoaderExtractor, _ttemp_flatten

import torch
import torchvision.transforms as transforms


class FashionMNIST_Dataloader(DataLoaderExtractor):
    
    def __init__(self, batch_size = 64, **kwargs):
        super().__init__(batch_size=batch_size)
        
        resize = transforms.Resize((28, 28))
        self.transform = transforms.Compose([
            resize,
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(_ttemp_flatten)
        ])
    
    def load_dataloader(self, download = True, split = None, **kwargs):
        dataset = FashionMNIST(
            './data',
            transform=self.transform,
            train=(split == "train"),
            download=download
        )
        
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        
    