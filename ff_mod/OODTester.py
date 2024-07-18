import datetime
from os import makedirs

import json

import logging as log

from ff_mod.metricable import get_basic_ood_metrics, save_auroc_curve
from ff_mod.dataloader.factory import DataloaderFactory

class OODTest:
    """
        Main class to launch the OOD tests.
        
        Different OOD score algorithms and datasets should be available
        
        Objectives:
            - Get AUROC, AUPR and FPR95
            - Save results
    """
    
    def __init__(self, trainer, name = "default", path = "logs/OOD/"):
        self.trainer = trainer
        
        # The save path should be in log/{number of total logs}_{date}/
        self.save_path = f"{path}/{name}/"
        print(self.save_path)
        makedirs(self.save_path)
        
        self.algorithms = []
        self.ood_datasets = []
        self.ood_dataset_params = []
        
        self.ID_scores_cache = {} # Saves the ID scores for each algorithm
        
    def add_algorithm(self, algorithm):
        self.algorithms.append(algorithm)
    
    def add_ood_dataset(self, dataset, dataset_params = {}):
        self.ood_datasets.append(dataset)
        self.ood_dataset_params.append(dataset_params)
    
    def launch_all(self, batch_size: int = 512, verbose: int = 0, path_pre: str = "SNN", **kwargs):      
        results = {}
        
        for algorithm in self.algorithms:
            algorithm.save_config(self.save_path)
        
        for ood_dataset_idx, ood_dataset in enumerate(self.ood_datasets):
            print("Testing dataset: " + ood_dataset)
            
            results[ood_dataset] = {}
            
            ood_dataloader = DataloaderFactory.get_instance().get(ood_dataset, batch_size=batch_size, **kwargs)
            ood_dataloader = ood_dataloader.get_dataloader(**self.ood_dataset_params[ood_dataset_idx])
                        
            for algorithm_idx, algorithm in enumerate(self.algorithms):
                print("Testing algorithm: " + algorithm.get_name())

                if algorithm_idx not in self.ID_scores_cache:
                    print("Getting ID scores")
                    ID_scores = algorithm.get_scores(self.trainer.test_loader, self.trainer.get_network(), **kwargs)
                    self.ID_scores_cache[algorithm_idx] = ID_scores
                else:
                    ID_scores = self.ID_scores_cache[algorithm_idx]
                    
                print("Getting OOD scores")
                OOD_scores = algorithm.get_scores(ood_dataloader, self.trainer.get_network(), **kwargs)
                
                scores, labels = ID_scores + OOD_scores, [0] * len(ID_scores) + [1] * len(OOD_scores)

                
                results[ood_dataset][algorithm.get_name()] = get_basic_ood_metrics(labels, scores)
                save_auroc_curve(labels, scores, f"{self.save_path}/{algorithm.get_name()}", ood_dataset)
        
        with open(self.save_path + "/" + 'results.json', 'w') as file:
            json.dump(results, file)
        
        return results