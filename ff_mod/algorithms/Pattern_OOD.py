from abc import abstractmethod
from ff_mod.algorithms.base import BaseOODAlgorithm

from ff_mod.networks.abstract_forward_forward import AbstractForwardForwardNet 

import torch

import logging as log

import os
import json

from tqdm import tqdm

from ff_mod.algorithms.distances import *

class PatternAlgorithm(BaseOODAlgorithm):
    def __init__(self):
        super().__init__()
        self.latent_spaces = None
    
    @torch.no_grad()
    @abstractmethod
    def initial_setup(self, net, in_loader, sample_size: int = 512):
        pass
    
    def get_name(self):
        pass
    
    def save_config(self, path, params):        
        if os.path.isfile(path + "/" + self.__class__.__name__ + ".json"):
            return
        
        with open(path + "/" + self.__class__.__name__ + ".json", 'w') as f:
            json.dump(params, f)

    @abstractmethod
    def get_scores_batch(self, data, net : AbstractForwardForwardNet, **kwargs):
        pass

    @torch.no_grad()
    def get_scores(self, ood_loader, net: AbstractForwardForwardNet, **kwargs):
        if self.latent_spaces is None:
            raise Exception("Initial patterns were not set. Please call initial_setup first.")
            
        scores = [] 
        for step, (x,y) in tqdm(enumerate(iter(ood_loader)), disable=self.verbose<1, total=len(ood_loader)):
            x, y = x.to(self.device), y.to(self.device)
            
            scores += self.get_scores_batch(x, net, **kwargs)
            
        return scores 


class PatternOODv3_Geo(PatternAlgorithm):
    
    # TODO Refactor so that dim is automatically inferred from the network
    def __init__(
            self,
            latent_depth: int = 1,
            p = 2,
            inverse_base = 10e4,
            zero_scale = 1.0,
            num_classes = 10,
            distance = "manhattan",
            device = "cpu",
            verbose = 1
        ):
        
        self.latent_depth = latent_depth

        
        self.distance = distance
        
        self.p = p
        
        self.device = device
        
        self.verbose = verbose
        
        self.latent_spaces = None
        self.num_classes = num_classes
        
        self.inverse_base = inverse_base
        self.zero_scale = zero_scale
        
    
    def __get_latent(self, x, fake_lab, net : AbstractForwardForwardNet):
        latents = net.get_latent(x, fake_lab, self.latent_depth, overlay_factor = 1)
        if len(latents.shape) == 3:
            latents = latents.mean(axis=1)
            
        return latents
    
    @torch.no_grad()
    def initial_setup(self, net, in_loader, sample_size: int = 1000, lower_bound = 0.2, higher_bound = 0.8):
        dim = net.layers[self.latent_depth].weight.shape[0]

        self.latent_spaces = [[[] for _ in range(self.num_classes)] for _ in range(self.num_classes)] 
        
        for step, (x, y) in  tqdm(enumerate(iter(in_loader)), disable=self.verbose<1):
            x, y = x.to(self.device), y.to(self.device)
            
            for j in range(0,self.num_classes):
                fake_lab = (y+j)%self.num_classes
                
                for i, latent in enumerate(self.__get_latent(x, fake_lab, net)):
                    self.latent_spaces[y[i]][fake_lab[i]] += [latent]
                
            if step * x.shape[0] > sample_size:
                break
        
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                
                latent = torch.stack(self.latent_spaces[i][j])
            
                if i==j:
                    latent = latent[torch.norm(latent, p=self.p, dim=1).sort()[1][int(latent.shape[0]*lower_bound):]]
                else:
                    latent = latent[torch.norm(latent, p=self.p, dim=1).sort()[1][:int(latent.shape[0]*higher_bound)]]
                    
                self.latent_spaces[i][j] = latent
            
    
    def get_name(self):
        return f"PatternOODv3_Geo_{self.distance}"

    def save_config(self, path):
        params = {'latent_depth': self.latent_depth, 'distance': self.distance}
        super().save_config(path, params)
    
    def get_minimal_distance(self, x, group, power_sum = False):
        
        if self.distance == "manhattan":
            dist_temp = torch.cdist(x, group, p=1).min(axis=1)[0]
        elif self.distance == "euclidean":
            dist_temp = torch.cdist(x, group, p=2).min(axis=1)[0]
        elif self.distance == "cosine":
            A_norm = x / (x.norm(dim=1, keepdim=True)+0.0001)
            B_norm = group / (group.norm(dim=1, keepdim=True)+0.0001)

            dist_temp = 1 - torch.mm(A_norm, B_norm.t()).min(axis=1)[0]

        return dist_temp
    
    @torch.no_grad()
    def get_scores_batch(self, data, net : AbstractForwardForwardNet, power_sum = False, **kwargs):
        
        latent_dist = torch.zeros((data.shape[0], self.num_classes)).to(self.device)
        zero_dist = torch.zeros((data.shape[0],)).to(self.device)
                
        for label in range(self.num_classes):
            latent_vec = self.__get_latent(data, torch.full((data.shape[0],), label, dtype=torch.long), net)
                
            for pos_class in range(self.num_classes):
                latent_dist[:, pos_class] += self.get_minimal_distance(latent_vec, self.latent_spaces[pos_class][label], power_sum=power_sum)
                
            zero_dist += self.get_minimal_distance(latent_vec, torch.zeros_like(latent_vec), power_sum=power_sum)
        
        zero_greatest = (self.zero_scale * zero_dist < latent_dist.min(axis=1)[0]).float()
        
        score_init = latent_dist.min(axis=1)[0]
        
        return ( score_init + (self.inverse_base - 2 * score_init)*zero_greatest).tolist()
        #return score_init.tolist()
    