from torchvision.datasets import Omniglot
from ff_mod.dataloader.base import DataLoaderExtractor, _ttemp_flatten

import torch
import torchvision.transforms as transforms


class OneMinusTransform__z(object):
    def __call__(self, x):
        return torch.ones_like(x) - x

class Omniglot_Dataloader(DataLoaderExtractor):
    
    def __init__(self, batch_size = 64, **kwargs):
        super().__init__(batch_size=batch_size)
        
        resize = transforms.Resize((28, 28))
        self.transform = transforms.Compose([
            resize,
            transforms.ToTensor(),
            OneMinusTransform__z(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(_ttemp_flatten)
        ])
    
    
    def load_dataloader(self, download = False, **kwargs):
        dataset = Omniglot(
            './data',
            background=True,
            transform=self.transform,
            download=download
        )
        
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )