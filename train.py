import argparse
import json
import os
from datetime import datetime


from ff_mod.networks.spiking_network import SpikingNetwork
from ff_mod.networks.base_network import AnalogNetwork

from ff_mod.dataloader.factory import DataloaderFactory

from ff_mod.overlay import AppendToEndOverlay
from ff_mod.loss.loss import VectorBCELoss

from ff_mod.trainer import Trainer

from ff_mod.callbacks.accuracy_writer import AccuracyWriter

from torch.utils.tensorboard.writer import SummaryWriter

from ff_mod.utils import save_experiment_info, create_network

EXPERIMENTAL_FOLDER = "experiments/train"

CURRENT_TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")


def train(network, args):
        trainer = Trainer(device = 'cuda:0', greedy_goodness = args["greedy_goodness"])
        trainer.load_data_loaders(args["dataset"], batch_size = args["batch_size"], test_batch_size = args["batch_size"], resize=(args["dataset_resize"], args["dataset_resize"]))
        trainer.set_network(network)

        experiment_name = f"{args['dataset']}_{'SNN' if args['use_snn'] else 'ANN'}_({args['neurons_per_layer']})_{CURRENT_TIMESTAMP}/"
        
        writer = SummaryWriter(f"{EXPERIMENTAL_FOLDER}/{experiment_name}/summary/" )
        
        trainer.add_callback(AccuracyWriter(tensorboard=writer))
        trainer.train(epochs=args["epochs"], verbose=1)
        network.save_network(f"{EXPERIMENTAL_FOLDER}/{experiment_name}/model")

def main():
    parser = argparse.ArgumentParser(description="Save experimental information to a JSON file")
    
    # Training Arguments
    parser.add_argument("--dataset", default="mnist", help="Dataset to use")
    parser.add_argument("--dataset_resize", type=int, default=28, help="Dataset to use")
    
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
    parser.add_argument("--greedy_goodness",default=False, action=argparse.BooleanOptionalAction, help="Whether to use greedy goodness")
    
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--use_snn", default=False, action=argparse.BooleanOptionalAction, help="Use spiking networks")
    
    # Overlay Arguments
    parser.add_argument("--pattern_size", type=int, default=100, help="Pattern size")
    parser.add_argument("--num_vectors", type=int, default=1, help="Number of vectors per class")
    parser.add_argument("--p", type=float, default=0.1, help="Percentaje of ones in the vectors")
    
    # Network Argument
    parser.add_argument("--neurons_per_layer", type=int, default=1000, help="Number of neurons per layer")
    parser.add_argument("--num_steps", type=int, default=20, help="Number of steps for the spiking network")
    parser.add_argument("--internal_epoch", type=int, default=10, help="Number of epochs for the spiking network")
    parser.add_argument("--save_activity", default=False, action=argparse.BooleanOptionalAction, help="Save activity of the network")
    parser.add_argument("--input_size", type=int, default=784, help="Input size of the network")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layer of the network")
    parser.add_argument("--bounded_goodness", default=False, action=argparse.BooleanOptionalAction, help="Activate bounded goodness in SNN.")
    
    # Loss Arguments
    parser.add_argument("--loss", default="VectorBCELoss", help="Loss function to use")
    parser.add_argument("--threshold", type=float, default=6, help="Threshold for the loss function")
    parser.add_argument("--alpha", type=float, default=1, help="Alpha for the loss function")
    parser.add_argument("--beta", type=float, default=1, help="Beta for the loss function")
    parser.add_argument("--negative_threshold", type=float, default=2, help="Negative threshold for the loss function")
    
    
    args = vars(parser.parse_args())

    save_experiment_info(args, EXPERIMENTAL_FOLDER, CURRENT_TIMESTAMP)
    
    network = create_network(args)
    train(network, args)

if __name__ == "__main__":
    main()
