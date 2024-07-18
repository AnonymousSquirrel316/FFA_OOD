from ff_mod.callbacks.callback import Callback
from sklearn.metrics import confusion_matrix

import seaborn as sns

class AccuracyWriter(Callback):
    def __init__(self, tensorboard = None) -> None:
        super().__init__()
        
        self.train_acc = 0.0
        self.test_acc = 0.0
        
        self.train_len = 0.0
        self.test_len = 0.0
        
        self.train_total_len = 0.0
        self.test_total_len = 0.0
                
        self.tensorboard = tensorboard
        
    def reset(self):
        self.train_acc = 0.0
        self.test_acc = 0.0
        
        self.train_len = 0.0
        self.test_len = 0.0
    
    def on_train_batch_end(self, *args, **kwargs):
        super().on_train_batch_end(*args, **kwargs)
        
        self.train_acc += kwargs["predictions"].eq(kwargs["y"]).float().sum()
        
        self.train_len += kwargs["y"].shape[0]
        self.tensorboard.add_scalar("train_acc_bacth", self.train_acc / self.train_len, self.train_total_len + self.train_len)
        
    def on_test_batch_end(self, *args, **kwargs):
        super().on_test_batch_end(*args, **kwargs)
        
        self.test_acc += kwargs["predictions"].eq(kwargs["y"]).float().sum()
        
        self.test_len += kwargs["y"].shape[0]
        self.tensorboard.add_scalar("test_acc_bacth", self.test_acc / self.test_len, self.test_total_len + self.test_len)
            
    def on_train_epoch_end(self, *args, **kwargs):
        super().on_train_epoch_end(*args, **kwargs)
        self.tensorboard.add_scalar("train_acc", self.train_acc / self.train_len, self.train_total_len + self.train_len)
        
        self.train_total_len += self.train_len
        
        self.train_acc = 0.0
        self.train_len = 0.0
        
    def on_test_epoch_end(self, *args, **kwargs):
        super().on_test_epoch_end(*args, **kwargs)
        self.tensorboard.add_scalar("test_acc", self.test_acc / self.test_len, self.test_total_len + self.test_len)
        
        self.test_total_len += self.test_len
        
        self.test_acc = 0.0
        self.test_len = 0.0
        