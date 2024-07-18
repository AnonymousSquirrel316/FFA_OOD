from ff_mod.callbacks.callback import Callback
from sklearn.metrics import confusion_matrix

import seaborn as sns

from statistics import mean

class LayerIntensityCallback(Callback):
    def __init__(self, net, tensorboard = None) -> None:
        super().__init__()

        self.net = net
        self.tensorboard = tensorboard
        self.ii = 0
        
    def reset(self):
        pass
    
    def on_train_batch_end(self, *args, **kwargs):
        super().on_train_batch_end(*args, **kwargs)
        for i, layer in enumerate(self.net.layers):
            self.tensorboard.add_scalar("layer_" + str(i)+'_average_intesity', mean(layer.activity_saver.avg_activity+ [0]), self.ii)
            
            self.tensorboard.add_scalar("layer_" + str(i)+'_positive_intesity', mean(layer.activity_saver.pos_activity+ [0]), self.ii)
            self.tensorboard.add_scalar("layer_" + str(i)+'_negative_intesity', mean(layer.activity_saver.neg_activity+ [0]), self.ii)
            self.tensorboard.add_scalar("layer_" + str(i)+'_residual_intesity', mean(layer.activity_saver.res_activity+ [0]),  self.ii)
        
        self.ii += 1
            
        