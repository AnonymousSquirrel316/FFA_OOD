import torch 
import numpy as np

import os

import matplotlib.pyplot as plt

from sklearn.metrics import auc, precision_recall_curve, roc_curve

def get_values_and_labels(trainer, overlay_function, ood_loader, verbose = 1):



    # Number of classes inside a torch dataloader 
    n_classes = len(torch.unique(trainer.test_loader.dataset.targets))

    scores = []
    labels = []
        
    for step, (x,y) in  enumerate(iter(trainer.test_loader)):
        x, y = x.cuda(), y.cuda()


        x = trainer.net.adjust(x)
        goodness_per_label = []
        for label in range(n_classes):
            h = overlay_function(x, torch.full((x.shape[0],), label, dtype=torch.long))
            goodness = []
            for layer in trainer.net.layers:
                h = layer(h)
                goodness += [h.mean(1).pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        
        sorted_x, _ = torch.sort(goodness_per_label, dim=1)
        
        scores += [sorted_x[:, 4]]
        labels += [1]
    
    for step, (x,y) in  enumerate(iter(ood_loader)):
        x, y = x.cuda(), y.cuda()


        x = trainer.net.adjust(x)
        goodness_per_label = []
        for label in range(n_classes):
            h = overlay_function(x, torch.full((x.shape[0],), label, dtype=torch.long))
            goodness = []
            for layer in trainer.net.layers:
                h = layer(h)
                goodness += [h.mean(1).pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        
        sorted_x, _ = torch.sort(goodness_per_label, dim=1)
        
        scores += [sorted_x[:, 4]]
        labels += [0]
    

    return scores, labels

def get_basic_ood_metrics(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    auroc = auc(fpr, tpr)
    
    
    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    aupr = auc(recall, precision)
    
    # Code from https://github.com/tayden/ood-metrics
    fpr95 = -1
    if all(tpr < 0.95):
        fpr95 = 0
    elif all(tpr >= 0.95):
        idxs = [i for i, x in enumerate(tpr) if x >= 0.95]
        fpr95 = min(map(lambda idx: fpr[idx], idxs))
    else:
        fpr95 = np.interp(0.95, tpr, fpr)
        
    return auroc, aupr, fpr95

def save_auroc_curve(labels, scores, save_path, algorithm_name = "FF_OOD"):
    
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    
    auroc = auc(fpr, tpr)
    
    plt.clf()
    
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (AUROC={:.4f})'.format(auroc))
    
    os.makedirs(save_path, exist_ok=True)
    
    plt.savefig(save_path+f"/{algorithm_name}.png")
    