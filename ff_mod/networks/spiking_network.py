from ff_mod.networks.abstract_forward_forward import AbstractForwardForwardNet, AbstractForwardForwardConvNet, AbstractForwardForwardLayer, AbstractForwardForwardLayerConv

import torch
from torch.optim import Adam

import snntorch as snn
from snntorch import surrogate

class EmptySpikingNetwork(AbstractForwardForwardNet):
    """Empty version of a Spiking Forward-Forward Network."""
    def __init__(
            self,
            overlay_function,
            first_prediction_layer = 0,
            save_activity = False,
            num_steps = 20
        ) -> None:
        super().__init__(overlay_function, first_prediction_layer, save_activity)
        self.num_steps = num_steps # For data adjustment
        self.setSpiking(True)
    
    def adjust_data(self, data):
        """
            Returns the data in the correct shape based on the Net properties.
            
            E.g. Spiking Neural Networks may need to adjust the data for temporal information.
        """
        return torch.reshape(data, [data.shape[0],1,data.shape[1]]).repeat(1, self.num_steps, 1)


class SpikingNetwork(AbstractForwardForwardNet):
    
    def __init__(
            self,
            overlay_function,
            loss_function,
            learning_rate = 0.002,
            dims = [784, 400, 400],
            num_steps = 20,
            internal_epoch = 10,
            first_prediction_layer = 0,
            save_activity = False,
            bounded_goodness = False
        ) -> None:
        super().__init__(overlay_function, first_prediction_layer, save_activity)
        
        self.setSpiking(True)
        
        self.num_steps = num_steps # For data adjustment
        
        for d in range(len(dims) - 1):
            layer = Layer(
                dims[d],
                dims[d + 1],
                loss_function,
                learning_rate,
                internal_epoch,
                num_steps,
                save_activity,
                bounded_goodness
            )
            self.layers.append(layer)
    
    def adjust_data(self, data):
        """
            Returns the data in the correct shape based on the Net properties.
            
            E.g. Spiking Neural Networks may need to adjust the data for temporal information.
        """
        return torch.reshape(data, [data.shape[0],1,data.shape[1]]).repeat(1, self.num_steps, 1)

class SpikingNetworkConv(AbstractForwardForwardConvNet):
    
    def __init__(
            self,
            overlay_function,
            loss_function,
            dims = [1, 50, 50],
            shapes = [[28,28], [12,12]],
            kernel=7,
            stride=2,
            num_steps = 20,
            internal_epoch = 10,
            first_prediction_layer = 0,
            save_activity = False,
        ) -> None:
        
        super().__init__(overlay_function, first_prediction_layer, save_activity)
        
        self.num_steps = num_steps
        self.internal_epoch = internal_epoch      
        
        for d in range(len(dims) - 1):
            self.layers.append(Conv2d(dims[d], dims[d + 1], kernel=kernel, stride=stride, internal_epoch = internal_epoch, num_steps = num_steps, loss_function = loss_function, save_activity=save_activity, dims=shapes[d]))
        
    
    def adjust_data(self, data):
        """
            Returns the data in the correct shape based on the Net properties.
            
            E.g. Spiking Neural Networks may need to adjust the data for temporal information.
        """
 
        if len(data.shape) == 2:
            data = torch.reshape(data, [data.shape[0], 28, 28])
            
        return torch.reshape(data, [data.shape[0],1, 1, data.shape[1], data.shape[2]]).repeat(1, self.num_steps, 1,1,1)
        
        

class Layer(AbstractForwardForwardLayer):
    def __init__(
            self,
            in_features,
            out_features,
            loss_function,
            learning_rate=0.002,
            internal_epoch = 10,
            num_steps = 20,
            save_activity = False,
            bounded_goodness = False
        ):
        super().__init__(in_features, out_features, loss_function, save_activity)
        
        threshold = 0.4 if in_features == 784 else 0.2
        beta_n = 0.4 if in_features == 784 else 0.3

        
        self.relu = snn.Leaky(beta=beta_n, threshold = threshold, init_hidden=True, spike_grad=surrogate.fast_sigmoid(), learn_beta=False, learn_threshold=False, reset_mechanism  = "zero")
        self.opt = Adam(self.parameters(), lr=learning_rate)
        
        self.counter = None
        
        self.num_epochs = internal_epoch
        self.num_steps = num_steps
        
        self.bounded_goodness = bounded_goodness

    def get_goodness(self, x):
        if self.bounded_goodness:
            return x.mean(1).mean(1)
        else:
            return x.sum(1).pow(2).mean(1)

    def forward(self, x, positivity_type = None):
        result = []
        
        self.relu.reset_hidden()
        self.counter = torch.ones(x.shape[0], self.out_features, requires_grad=False).to(x.device)
        
        for t in range(self.num_steps):
            bias = 0 if self.bias is None else self.bias.unsqueeze(0)
            
            res = self.relu(torch.mm(x[:,t,:], self.weight.T) * self.counter + bias)
            self.counter = self.counter * (1 - 0.27 * res)
            
            result += [res.reshape([x.shape[0], 1, self.weight.T.shape[1]])]
        
        result = torch.cat(result, dim=1)
        
        self.add_activity(result, positivity_type)
        
        return result
    


class Conv2d(AbstractForwardForwardLayerConv):
    def __init__(self, in_features, out_features, kernel, stride,loss_function,
                bias=False, device=None, dtype=None, internal_epoch = 10, num_steps = 20, save_activity = False, dims=None):
        super().__init__(in_features, out_features, kernel, stride, loss_function, bias, device, dtype, save_activity=save_activity)
        
        if in_features == 784:
            beta_n = 0.4
            threshold = 0.4
        else:
            beta_n = 0.3
            threshold = 0.2
            
        self.dims = dims
        
        # Tiempo de decay de la actividad
        # A ver si da tiempo a que se descarge por completo en tiempo de simulacion
        self.relu = snn.Leaky(beta=beta_n, threshold = threshold, init_hidden=True, spike_grad=surrogate.fast_sigmoid(), learn_beta=False, learn_threshold=False, reset_mechanism  = "zero")
        self.opt = Adam(self.parameters(), lr=0.005)
        #self.bn = torch.nn.BatchNorm1d(in_features)
        self.threshold = 1.0
        self.num_epochs = internal_epoch
        self.num_steps = num_steps

    def get_goodness(self, x):
        #return (1/(1-x.mean(1).pow(2)+0.033) - 1).mean(1)
        #return -torch.log(1-x.mean(1) + 0.033).mean(1)
        ppp = 5 * x.mean(1).mean(1)
        return ppp
        #return (15*torch.nn.functional.relu(x-0.3)).mean(1).square().mean(1)

    def forward(self, x, positivity_type = None, label_vector=None):
        # X should have a shape of [batch, num_steps, dim]
        # x_direction = x / (x.norm(2, 2, keepdim=True) + 1e-4)
        # I think normalization like this won't workç
        
        
        # Asume no bias
        result = []
        
        self.relu.reset_hidden()
        
        for t in range(self.num_steps):
            x_reshaped = x[:,t,:].view(x.shape[0], self.in_channels, self.dims[0], self.dims[1])

            x_reshaped = torch.nn.functional.conv2d(x_reshaped, self.weight, stride=self.stride) # Shape [Batch, out_channels, heigh , width]
            
            expanded_binary_vector = label_vector.view(x.shape[0], self.out_channels, 1, 1)
   
            x_reshaped = x_reshaped * expanded_binary_vector

            result += [self.relu(x_reshaped).reshape([x.shape[0], 1, x_reshaped.shape[1]*x_reshaped.shape[2]*x_reshaped.shape[3]])]
        
        result = torch.cat(result, dim=1)
        self.add_activity(result, positivity_type)
        
        return result