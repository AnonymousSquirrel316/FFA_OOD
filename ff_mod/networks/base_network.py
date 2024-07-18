import torch
from torch.optim import Adam

from ff_mod.networks.abstract_forward_forward import AbstractForwardForwardNet, AbstractForwardForwardLayer

from ff_mod.overlay import Overlay

class AnalogNetwork(AbstractForwardForwardNet):

    def __init__(
            self,
            overlay_function : Overlay,
            dims,
            loss_function,
            learning_rate = 0.001,
            internal_epoch = 20,
            first_prediction_layer = 0,
            save_activity = False
        ):
        
        super().__init__(overlay_function, first_prediction_layer, False)
        
        self.internal_epoch = internal_epoch
        
        for d in range(len(dims) - 1):
            layer = Layer(dims[d], dims[d + 1], loss_function, learning_rate, save_activity, internal_epoch)
            self.layers.append(layer)

        self.layers[0].is_input_layer = True
        
    def adjust_data(self, data):
        return data

class Layer(AbstractForwardForwardLayer):
    def __init__(
            self,
            in_features,
            out_features,
            loss_function,
            learning_rate = 0.001,
            save_activity = False,
            internal_epoch = 10,
            **kwargs
        ):
        super().__init__(in_features, out_features, loss_function = loss_function, save_activity = save_activity, **kwargs)
        
        self.is_input_layer = False
        
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(in_features)
        
        self.opt = Adam(self.parameters(), lr=learning_rate)

        self.num_epochs = internal_epoch

    def get_goodness(self, x):
        return x.pow(2).mean(1)
    
    def forward(self, x, positivity_type = None):
        
        bias_term = 0
        if self.bias is not None:
            bias_term = self.bias.unsqueeze(0)
        
        #if not self.is_input_layer:
        #    x = self.bn(x)

        result =  self.relu(torch.mm(x, self.weight.T) + bias_term)
            
        self.add_activity(result, positivity_type)
        
        return result
