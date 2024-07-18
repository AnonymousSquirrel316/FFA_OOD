from torchvision.datasets import MNIST

from ff_mod.dataloader.base import DataLoaderExtractor, _ttemp_flatten, _base_normalization

import torch
import torchvision.transforms as transforms


class MNIST_Dataloader(DataLoaderExtractor):
    
    def __init__(self, batch_size = 64, **kwargs):
        super().__init__(batch_size=batch_size)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(_ttemp_flatten)
        ])
    
    def load_dataloader(self, download = True, split = None, subset = None, **kwargs):
        dataset = MNIST(
            './data',
            transform=self.transform,
            train=(split == "train"),
            download=download
        )
        
        if subset is not None:
            idx = torch.tensor([label in subset for label in dataset.targets])

            dataset.targets = dataset.targets[idx]
            dataset.data = dataset.data[idx]
            
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        