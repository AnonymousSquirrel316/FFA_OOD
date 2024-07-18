from ff_mod.OODTester import OODTest
from ff_mod.trainer import Trainer

from ff_mod.overlay import AppendToEndOverlay

from ff_mod.networks.prebuilt.snn import A1_LDS as A1_LDS_SNN
from ff_mod.networks.prebuilt.ann import A1_LDS_ANN as A1_LDS_ANN

from ff_mod.dataloader.factory import DataloaderFactory

from ff_mod.algorithms.Pattern_OOD import PatternOODv3_Geo
from ff_mod.loss.loss import VectorBCELoss

from ff_mod.utils import save_experiment_info_ood, create_network

from datetime import datetime

import argparse
import json

import json

import torch

import os



EXPERIMENTAL_FOLDER_OOD = "experiments/OOD"
EXPERIMENTAL_FOLDER_TRAIN = "experiments/train"

experiment_path = ""

CURRENT_TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

def get_experiment_list(experiment_dataset_list, use_snn = True):
    experiments = []

    for f in os.listdir(EXPERIMENTAL_FOLDER_TRAIN):
        with open(f"{EXPERIMENTAL_FOLDER_TRAIN}/{f}/experiment_data.json", 'r') as file:
            experiment_data = json.load(file)
            
        if "train" not in f:
            experiment_data['bounded_goodness'] = True
        else:
            experiment_data['bounded_goodness'] = False
        
        
        if experiment_data['dataset'] in experiment_dataset_list:
            experiments.append(f)
        
    return experiments


def launch_ood_tests(experiments, ood_group_datasets, args):
    global experiment_path
    for i, experiment in enumerate(experiments):
        
        # Get experiment data
        experiment_data = None
        with open(f"{EXPERIMENTAL_FOLDER_TRAIN}/{experiment}/experiment_data.json", 'r') as file:
            experiment_data = json.load(file)
        
        # Create network
        network = create_network(experiment_data)
        network.cuda()
        
        network.load_network(f"{EXPERIMENTAL_FOLDER_TRAIN}/{experiment}/model")
        
        # Create and load trainer
        trainer = Trainer()
        trainer.set_network(network)
        trainer.load_data_loaders(experiment_data['dataset'], batch_size = args['batch_size'], test_batch_size=args['batch_size'], resize=args['resize'])

        # Create OOD test
        ood_test = OODTest(trainer=trainer, name=experiment, path = experiment_path)
        
        # Add OOD datasets
        for dataset in ood_group_datasets:
            if dataset != experiment_data['dataset']:
                ood_test.add_ood_dataset(dataset, {"split": "test"})
            
        # Add algorithms
        distances = [["manhattan", None], ["euclidean", None], ["cosine", None]]
        for distance in distances:
            
            num_classes = DataloaderFactory.get_instance().get_num_classes(experiment_data['dataset'])
            
            pattern_oodv3_geo = PatternOODv3_Geo(distance = distance[0], device="cuda:0", latent_depth=1, zero_scale=args['zero_scale'], inverse_base=args['inverse_base'], num_classes=num_classes, verbose=1)
            pattern_oodv3_geo.initial_setup(trainer.get_network(), trainer.train_loader)
        
            ood_test.add_algorithm(pattern_oodv3_geo)
            
            torch.cuda.empty_cache()

        # TODO Make variable resize inside DataloaderFactory
        ood_test.launch_all(verbose=1, path_pre=experiment, resize=args['resize'])
        
    
def main():
    global experiment_path
    
    parser = argparse.ArgumentParser(description="Save experimental information to a JSON file")
    
    # Training Arguments
    parser.add_argument("--in_dataset", default=None, help="In distribution dataset. If all, all datasets in group will be used")
    parser.add_argument("--dataset_group", default = "small", choices=["small"], help="Choose OOD dataset group. Small(28x28))")
    
    parser.add_argument("--use_snn", default=False, action=argparse.BooleanOptionalAction, help="")
    
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")    
    parser.add_argument("--device", default="cuda:0", help="Device to use")
      
    parser.add_argument("--inverse_base", default=10e4, type = float, help="")
    parser.add_argument("--zero_scale", default=1, type = float, help="")
    parser.add_argument("--power_sum", default=False, action=argparse.BooleanOptionalAction, help="")
    
    args = vars(parser.parse_args())

    
    print("A")
    
    if args['in_dataset'] is not None:
        experiment_name = f"{args['in_dataset']}_{CURRENT_TIMESTAMP}/"
    else:
        experiment_name = f"{args['dataset_group']}_{CURRENT_TIMESTAMP}/"
    experiment_path = f"{EXPERIMENTAL_FOLDER_OOD}/{experiment_name}/"
    save_experiment_info_ood(args, experiment_path)
    
    # Get ood dataset list
    if args["dataset_group"] == "small":
        resize = (28, 28)
        ood_datasets = DataloaderFactory.get_instance().get_valid_dataloaders()
    
    args["resize"] = resize
    
    if args["in_dataset"] is None:
        trained_datasets = ood_datasets.copy()
    else:
        trained_datasets = [args["in_dataset"]]

    experiment_list = get_experiment_list(trained_datasets, use_snn=args['use_snn'])
    
    print(experiment_list)
    launch_ood_tests(experiment_list, ood_datasets, args)
    
if __name__ == "__main__":
    main()
