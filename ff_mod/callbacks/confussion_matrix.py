from ff_mod.callbacks.callback import Callback
from sklearn.metrics import confusion_matrix

import seaborn as sns

class ConfussionMatrixCallback(Callback):
    def __init__(self, n_classes = 10, tensorboard = None) -> None:
        super().__init__()
        
        self.n_classes = n_classes
        self.confussion_matrix = None
        
        self.tensorboard = tensorboard
        
    def reset(self):
        self.matrix = None
    
    def on_train_batch_end(self, *args, **kwargs):
        super().on_train_batch_end(*args, **kwargs)
        
        if self.confussion_matrix is None:
            self.confussion_matrix = confusion_matrix(kwargs["predictions"], kwargs["y"], labels=range(self.n_classes))
        else:
            self.confussion_matrix += confusion_matrix(kwargs["predictions"], kwargs["y"], labels=range(self.n_classes))
            
    def on_train_epoch_end(self, *args, **kwargs):
        if self.tensorboard is None:
            raise ValueError("No tensorboard was assigned")
        if self.confussion_matrix is None:
            raise ValueError("The confussion matrix was never created")
        
        heatmap = sns.heatmap(self.confussion_matrix, cbar=False)
    
        fig = heatmap.get_figure()
        self.tensorboard.add_figure("confussion_matrix", fig, self.train_step)
        