from abc import abstractmethod

from ff_mod.networks.abstract_forward_forward import AbstractForwardForwardNet

class BaseOODAlgorithm:
    
    def __init__(self):
        pass
    
    @abstractmethod
    def initial_setup(self, net, in_loader):
        pass
    
    @abstractmethod
    def save_config(self, path):
        pass
    
    @abstractmethod
    def get_scores(self, ood_loader, net : AbstractForwardForwardNet):
        pass
    
    def __call__(ood_loader, net : AbstractForwardForwardNet):
        """
        Executes the OOD algorithm on the given dataloader.

        Args:
            dataloader (_type_): _description_
        
        Returns:
            scores [list]: List of scores for each sample in the dataloader
        
        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()
        