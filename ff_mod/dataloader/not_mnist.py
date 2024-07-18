from ff_mod.dataloader.base import DataLoaderExtractor

import deeplake

class NotMNIST(DataLoaderExtractor):
    
    def __init__(self, batch_size = 64, large = False, **kwargs):
        super().__init__(batch_size=batch_size)
        self.version = "small" if not large else "large"
    
    def load_dataloader(self, download = False, split = None, **kwargs):
        ds = deeplake.load('hub://activeloop/not-mnist-' + self.version)
        
        self.dataloader = ds.pytorch(num_workers=0, batch_size=self.batch_size, shuffle=True, transform = {'images': self.transform, 'labels': None})
    
        