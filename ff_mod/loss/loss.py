from typing import Optional
import torch


class ProbabilityBCELoss:
    """
        BCE Loss for probability values of the goodness vector
    """
    def __init__(self):
        super().__init__()
        
    def __call__(self, g_pos, g_neg, **kwargs):
        return -torch.log(torch.cat([g_pos, (1-g_neg)])).mean()



class VectorBCELoss:
    def __init__(
            self,
            threshold = 0.5,
            alpha = 1.0,
            beta: Optional[float] = None,
            negative_threshold : Optional[float] = None
        ):
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        
        self.threshold = threshold
        self.negative_threshold = negative_threshold
        
        if beta is None:
            self.beta = alpha
        if negative_threshold is None:
            self.negative_threshold = threshold
    
    def __call__(self, g_pos, g_neg, **kwargs):
        
        loss = torch.nn.functional.softplus(torch.cat([
                    self.alpha*(-g_pos + self.threshold),
                    self.beta*(g_neg - self.negative_threshold)])).mean()
        return loss

class MeanBCELoss:
    def __init__(self, threshold: float = 0.5, alpha = 1.0, beta: Optional[float] = None):
        super().__init__()
        
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        
        if beta is None:
            self.beta = alpha
        
    def __call__(self, g_pos, g_neg, **kwargs):
        loss_pos = torch.log(1 + torch.exp(self.alpha * (-g_pos.mean() + self.threshold)))
        loss_neg = torch.log(1 + torch.exp(self.beta * (g_neg.mean() - self.threshold)))
        return loss_pos+loss_neg

class SymbaLoss:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, g_pos, g_neg, **kwargs):
        return torch.nn.functional.softplus(self.alpha * (g_neg.mean() - g_pos.mean()))
    
    
class SwymbaLoss:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, g_pos, g_neg, **kwargs):
        G = g_neg.mean() - g_pos.mean()
        
        return torch.nn.functional.silu(self.alpha * G)

class ContrativeFF:
    
    def __init__(self, threshold = 0.5, alpha = 1.0, beta : Optional[float] = None, ratio = 0.1) -> None:
        self.base_loss = VectorBCELoss(threshold = threshold, alpha = alpha, beta = beta)
        
        self.ratio = ratio
    
    def __call__(self, g_pos, g_neg, latent_pos = None, latent_neg = None, labels = None, **kwargs):
        
        if len(latent_pos.shape) == 3:
            latent_pos = latent_pos.mean(dim=1)
        
        # Sum of pairwise distance between different classes
        dists = torch.cdist(latent_pos, latent_pos, p=2)/latent_pos.shape[1]

        labels = labels.view(-1, 1)
        mask = labels != labels.t()

        dists_masked = dists * mask.float()
        
        dists_masked = torch.log(1 + torch.exp(-dists_masked))
        dists_masked *= mask.float() # Remove the same label pairs
        
        loss1 = dists_masked.mean()
        
        loss2 = self.base_loss(g_pos, g_neg)
        
        return self.ratio * loss1 + loss2