from ff_mod.algorithms.base import BaseOODAlgorithm

from ff_mod.networks.abstract_forward_forward import AbstractForwardForwardNet 

import torch

import os
import json

class FF_OOD(BaseOODAlgorithm):
    
    def __init__(self, choice_index = 4):
        self.choice_index = choice_index
    
    def initial_setup(self, net, in_loader):
        pass
    
    def get_name(self):
        return "FF_OOD"
    
    def save_config(self, path):
        params = {'choice_index': self.choice_index}
        
        # Check if file exists
        if os.path.isfile(path + "/" + self.__class__.__name__ + ".json"):
            return
        
        # Create json file
        with open(path + "/" + self.__class__.__name__ + ".json", 'w') as f:
            json.dump(params, f)
    
    def get_scores(self, loader, net : AbstractForwardForwardNet, n_classes: int, **kwargs):
        """
        
        Args:
        
            n_classes: n_classes of the ID dataset
        
        """        
        scores = []
        
        for step, (x,y) in  enumerate(iter(loader)):
 
            x, y = x.cuda(), y.cuda()
            x = net.adjust_data(x)
            
            goodness_per_label = []
            for label in range(n_classes):
                
                h = net.overlay_function(x, torch.full((x.shape[0],), label, dtype=torch.long))
                goodness = []
                
                for layer in net.layers:
                    h = layer(h)
                    goodness += [layer.get_goodness(h)]
                
                goodness_per_label += [sum(goodness).unsqueeze(1)]
            
            goodness_per_label = torch.cat(goodness_per_label, 1)
            
            goodness_per_label, _ = torch.sort(goodness_per_label, dim=1)
            
            scores += goodness_per_label[:, self.choice_index].clone().detach().cpu().tolist()
            
        return scores 