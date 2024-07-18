import torch
import torch.nn as nn
import torch.nn.functional as F

from ff_mod.overlay import Overlay

from abc import ABC, abstractmethod

class ActivitySaver:
    def __init__(self) -> None:
        self.avg_activity = []
        self.pos_activity = []
        self.neg_activity = []
        self.res_activity = []
        
    def reset_activity(self):
        self.avg_activity = []
        self.pos_activity = []
        self.neg_activity = []
        self.res_activity = []
        
    def add_activity(self, activity, positivity_type = None):
        
        self.avg_activity += [activity]
        
        if positivity_type is not None:
            if positivity_type == "pos":
                self.pos_activity += [activity]
            elif positivity_type == "neg":
                self.neg_activity += [activity]
        else:
            self.res_activity += []

class AbstractForwardForwardNet(ABC, nn.Module):
    """
    Abstract class representing a FeedForward Forward-Forward Neural Network.
    
    Ref:
        - The Forward-Forward Algorithm: Some Preliminary Investigations - G. Hinton (https://arxiv.org/pdf/2212.13345.pdf)
    """
    
    def __init__(
            self,
            overlay_function : Overlay,
            first_prediction_layer = 0,
            save_activity = False
        ):
        """
        Args:
            overlay_function (_type_): _description_
            num_classes (_type_): Number of classes of the dataset.
            save_activity (bool, optional): True if the activity state of the network should be stored. Defaults to False.
        """
        super().__init__()
        self.overlay_function = overlay_function
        
        self.first_prediction_layer = first_prediction_layer
        
        self._save_activity = save_activity
        
        self.layers = torch.nn.ModuleList()
        
        self.is_spiking = False
        
    def add_layer(self, layer : nn.Module):
        self.layers.append(layer)
    
    def setSpiking(self, is_spiking):
        self.is_spiking = is_spiking
    
    @abstractmethod
    def adjust_data(self, data):
        """
        Adjust the data based on the network's properties.
        This method needs to be implemented by a subclass.
        """
        
    def train_network(self, x_pos, x_neg, labels = None):
        """
        Train the network based on the positive and negative examples.
        This method needs to be implemented by a subclass.
        """
        
        x_pos = self.adjust_data(x_pos)
        x_neg = self.adjust_data(x_neg)
        
        for i, layer in enumerate(self.layers):
            x_pos, x_neg = layer.train_network(x_pos, x_neg, labels = labels)
    
    @torch.no_grad()
    def predict(self, x, total_classes):
        """
        Predict the labels of the input data.

        Args:
            x (torch.Tensor): Input data. 
                Shape: [batch_size, input_size] if the network is not spiking, [batch_size, num_steps, input_size] otherwise.

        Returns:
            torch.Tensor: Predicted labels.
        """
        goodness_scores = []
        
        for label in range(total_classes):
            h = self.overlay_function(x, torch.full((x.shape[0],), label, dtype=torch.long))
            h = self.adjust_data(h)
            
            goodness = torch.zeros(h.shape[0], 1).to(x.device)
            for j, layer in enumerate(self.layers):
                h = layer(h)
                if j >= self.first_prediction_layer:
                    goodness += layer.get_goodness(h).unsqueeze(1)
            
            goodness_scores += [goodness]
        
        return torch.cat(goodness_scores, 1).argmax(1)
        
    def reset_activity(self):
        pass

    def get_latent(
            self,
            x,
            label,
            depth,
            unsupervised = False,
            overlay_factor = 1
        ):
        """Get the latent activation of the network at certain depth.

        Args:
            x (Torch.Tensor): Input data.
            label (_type_): Labels of the input data.
            depth (_type_): Depth of the latent activation.
            unsupervised (bool, optional): _description_. Defaults to False.
            overlay_factor (int, optional): _description_. Defaults to 1.

        Raises:
            ValueError: If depth is greater than the number of layers.

        Returns:
            _type_: _description_
        """
        if depth > len(self.layers):
            raise ValueError("Depth should not be greater than the number of layers")

        if not unsupervised:
            h = self.overlay_function(x, label)
            h[:, 784:] *= overlay_factor # TODO Add auto selection
        else:
            h = x.clone().detach()
        
        h = self.adjust_data(h)

        for i, layer in enumerate(self.layers):
            if i == depth:
                break

            h = layer(h)
            
        return h
    
    def save_network(self, path):
        torch.save(self.state_dict(), path)
        self.overlay_function.save(path + "_overlay_function")
    
    def load_network(self, path):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.overlay_function.load(path + "_overlay_function")
    

class AbstractForwardForwardLayer(ABC, nn.Linear):
    """
    Abstract class representing a layer in a Feed-Forward Neural Network.
    """
    def __init__(
            self,
            in_features,
            out_features,
            loss_function,
            bias=False,
            device="cuda:0",
            dtype=None,
            save_activity = False
        ):
        super().__init__(in_features, out_features, bias, device, dtype)
        
        self.save_activity = save_activity
        self.activity_saver = ActivitySaver()
        
        self.loss_function = loss_function
        
    
    def add_activity(self, result, positivity_type = None):
        if not self.save_activity:
            return
        
        self.activity_saver.add_activity(float(result.mean().clone().detach().cpu()), positivity_type)
    
    def reset_activity(self):
        self.activity_saver.reset_activity()
    
    @abstractmethod
    def get_goodness(self, x):
        pass
    
    @abstractmethod
    def forward(self, x, positivity_type = None):
        pass
    
    def train_network(self, x_pos, x_neg, labels = None):
        y, y_rnd = labels[0], labels[1]
        
        for epoch in range(self.num_epochs):
            self.opt.zero_grad()
            
            latent_pos = self.forward(x_pos, positivity_type="pos")
            g_pos = self.get_goodness(latent_pos)
            
            latent_neg = self.forward(x_neg, positivity_type="neg")
            g_neg = self.get_goodness(latent_neg)

            loss = self.loss_function(g_pos, g_neg, latent_pos = latent_pos, latent_neg = latent_neg, labels = y)
            loss.backward()
            self.opt.step()

        with torch.no_grad():
            return self.forward(x_pos).detach(), self.forward(x_neg).detach()
        

class AbstractForwardForwardConvNet(ABC, nn.Module):
    """
    Abstract class representing a FeedForward Forward-Forward Neural Network.
    
    Ref:
        - The Forward-Forward Algorithm: Some Preliminary Investigations - G. Hinton (https://arxiv.org/pdf/2212.13345.pdf)
    """
    
    def __init__(self, overlay_function, first_prediction_layer = 0, save_activity = False):
        """
        Args:
            overlay_function (_type_): _description_
            num_classes (_type_): Number of classes of the dataset.
            save_activity (bool, optional): True if the activity state of the network should be stored. Defaults to False.
        """
        super().__init__()
        self.overlay_function = overlay_function
        
        self.first_prediction_layer = first_prediction_layer
        
        self._save_activity = save_activity
        
        self.layers = torch.nn.ModuleList()
        
    
    @abstractmethod
    def adjust_data(self, data):
        """
        Adjust the data based on the network's properties.
        This method needs to be implemented by a subclass.
        """
        
    def train_network(self, x_pos, x_neg, labels = None):
        """
        Train the network based on the positive and negative examples.
        This method needs to be implemented by a subclass.
        """
        
        x_pos = self.adjust_data(x_pos)
        x_neg = self.adjust_data(x_neg)
        
        pos_label_vector = self.overlay_function.label_vectors[labels[0]]
        neg_label_vector = self.overlay_function.label_vectors[labels[1]]
   
        for i, layer in enumerate(self.layers):
                       
            x_pos, x_neg = layer.train_network(x_pos, x_neg, labels = labels, label_vector=[pos_label_vector, neg_label_vector])

    
    @torch.no_grad()
    def predict(self, x, total_classes):
        """
        Predict the labels of the input data.

        Args:
            x (torch.Tensor): Input data. 
                Shape: [batch_size, input_size] if the network is not spiking, [batch_size, num_steps, input_size] otherwise.

        Returns:
            torch.Tensor: Predicted labels.
        """
        # TODO Assert first_prediction_layer < len(self.layers)
        goodness_scores = []
        
        for label in range(total_classes):            
            label_vector = self.overlay_function.label_vectors[torch.full((x.shape[0],), label, dtype=torch.long)]
            h = self.adjust_data(x)
            
            goodness = torch.zeros(h.shape[0], 1).to(x.device)
            for j, layer in enumerate(self.layers):
                h = layer(h, label_vector=label_vector)
                if j >= self.first_prediction_layer:
                    goodness += layer.get_goodness(h).unsqueeze(1)
            
            goodness_scores += [goodness]
        
        return torch.cat(goodness_scores, 1).argmax(1)
        
    def reset_activity(self):
        #for layer in self.layers:
        #    layer.reset_activity()
        pass

class AbstractForwardForwardLayerConv(ABC, nn.Conv2d):
    """
    Abstract class representing a layer in a Feed-Forward Neural Network.
    """
    def __init__(
            self,
            in_features,
            out_features,
            kernel,
            stride,
            loss_function,
            bias=False,
            device="cuda:0",
            dtype=None,
            save_activity = False
        ):
        super().__init__(in_features, out_features, kernel_size=kernel, stride=stride,bias= bias,device= device,dtype= dtype)
        
        self.save_activity = save_activity
        self.activity_saver = ActivitySaver()
        
        self.loss_function = loss_function
        
    
    def add_activity(self, result, positivity_type = None):
        if not self.save_activity:
            return
        
        self.activity_saver.add_activity(float(result.mean().clone().detach().cpu()), positivity_type)
    
    def reset_activity(self):
        self.activity_saver.reset_activity()
    
    @abstractmethod
    def get_goodness(self, x):
        pass
    
    @abstractmethod
    def forward(self, x, positivity_type = None):
        pass
    
    def train_network(self, x_pos, x_neg, labels = None, label_vector=None):
        pos_label_vector, neg_label_vector = label_vector[0], label_vector[1]
        for epoch in range(self.num_epochs):
            self.opt.zero_grad()
            
            latent_pos = self.forward(x_pos, positivity_type="pos", label_vector=pos_label_vector)
            g_pos = self.get_goodness(latent_pos)
            
            
            latent_neg = self.forward(x_neg, positivity_type="neg", label_vector=neg_label_vector)
            g_neg = self.get_goodness(latent_neg)

            loss = self.loss_function(g_pos, g_neg, latent_pos = latent_pos, latent_neg = latent_neg, labels = labels)
            loss.backward()
            self.opt.step()

        with torch.no_grad():
            return self.forward(x_pos, label_vector=pos_label_vector).detach(), self.forward(x_neg, label_vector=neg_label_vector).detach()