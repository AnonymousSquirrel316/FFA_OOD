import torch
import torch.nn.functional as F

import numpy as np

def jaccard_similarity(pred, target):
    intersection = torch.sum(torch.min(pred, target.unsqueeze(0)), dim=1)
    union = torch.sum(torch.max(pred, target.unsqueeze(0)), dim=1)
    jaccard = (intersection+0.00001) / (union+0.00001)
    return jaccard

def get_vector_distance(tensor_sum, dist_name = "manhattan", binarize = 0.0):
    tensot_sum = tensor_sum.clone().detach()
        
    if binarize > 0.0:
        tensot_sum = (tensot_sum > binarize).float().clone().detach()
        
    dist = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            if dist_name == "manhattan":
                dist[i, j] = (tensot_sum[i]-tensot_sum[j]).abs().mean(dim=1)
            elif dist_name == "inv_manhattan":
                dist[i, j] = (1/(tensot_sum[i]+0.2)-1/(tensot_sum[j]+0.5)).abs().mean(dim=1)
            elif dist_name == "euclidean":
                dist[i, j] = torch.norm(tensot_sum[i] - tensot_sum[j], p=2, dim=1)
            elif dist_name == "cosine":
                dist[i, j] = 1 - F.cosine_similarity(tensot_sum[i], tensot_sum[j], dim=1)
            elif dist_name == "jacard":
                dist[i, j] = 1 - jaccard_similarity(tensot_sum[i], tensot_sum[j])
            elif dist_name == "spec":
                dist[i,j] = (tensot_sum[j] - torch.sign(tensot_sum[i]) * tensot_sum[j] + (1-torch.sign(tensot_sum[j])) * tensot_sum[i]).mean(dim=1)
    return dist

def get_distance(a, b, dist_name = "manhattan"):
    if dist_name == "manhattan":
        return (a-b).abs().mean(dim=1)
    elif dist_name == "inv_manhattan":
        return (1/(a+0.2)-1/(b+0.2)).abs().mean(dim=1)
    elif dist_name == "euclidean":
        return torch.norm(a - b, p=2, dim=1)
    elif dist_name == "cosine":
        return 1 - F.cosine_similarity(a, b, dim=1)
    elif dist_name == "jacard":
        return  1 - jaccard_similarity(a, b)
    elif dist_name == "spec":
        return (b - torch.sign(a) * b + (1-torch.sign(b)) * a).mean(dim=1)
