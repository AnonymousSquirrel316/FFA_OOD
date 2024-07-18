from ff_mod.networks.spiking_network import SpikingNetwork
from ff_mod.networks.base_network import AnalogNetwork

from ff_mod.dataloader.factory import DataloaderFactory
from ff_mod.overlay import AppendToEndOverlay

from ff_mod.loss.loss import VectorBCELoss

import os
import json

def save_experiment_info(data, experimental_folder, timestamp):
    experiment_name = f"{data['dataset']}_{'SNN' if data['use_snn'] else 'ANN'}_({data['neurons_per_layer']})_{timestamp}/"
    
    os.makedirs(experimental_folder, exist_ok=True)
    os.makedirs(f"{experimental_folder}/{experiment_name}/", exist_ok=True)

    filename = f"experiment_data.json"
    
    file_path = os.path.join(f"{experimental_folder}/{experiment_name}/", filename)

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def save_experiment_info_ood(data, experimental_folder):
    
    os.makedirs(experimental_folder, exist_ok=True)

    filename = f"experiment_data.json"
    file_path = os.path.join(experimental_folder, filename)

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def get_loss_function(args):
    if(args['loss'] == "VectorBCELoss"):
        return VectorBCELoss(args["threshold"], args["alpha"], args["beta"], args["negative_threshold"])

def create_network(args):
    layer_neurons = args["neurons_per_layer"]
    
    num_classes = DataloaderFactory.get_instance().get_num_classes(args["dataset"])
    
    overlay = AppendToEndOverlay(args["pattern_size"], num_classes, args["num_vectors"], p=args["p"])
    loss = get_loss_function(args)
    
    if args["use_snn"]:
        network = SpikingNetwork(
            overlay,
            loss,
            learning_rate=args["lr"],
            dims = [args["input_size"] + args["pattern_size"]] + [layer_neurons for _ in range(args['num_layers'])],
            num_steps = args["num_steps"],
            internal_epoch = args["internal_epoch"],
            bounded_goodness=args["bounded_goodness"])
    else:
        network = AnalogNetwork(
            overlay_function=overlay,
            loss_function=loss,
            learning_rate=args["lr"],
            dims= [args["input_size"] + args["pattern_size"]] + [layer_neurons for _ in range(args['num_layers'])],
            internal_epoch = args["internal_epoch"])

    if(args["device"] == "cuda:0"):
        network.cuda()

    return network

