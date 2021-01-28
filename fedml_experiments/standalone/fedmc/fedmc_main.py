import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
from fedml_api.data_preprocessing.fed_cifar100.data_loader import load_partition_data_federated_cifar100
from fedml_api.data_preprocessing.shakespeare.data_loader import load_partition_data_shakespeare
from fedml_api.data_preprocessing.fed_shakespeare.data_loader import load_partition_data_federated_shakespeare
from fedml_api.data_preprocessing.stackoverflow_lr.data_loader import load_partition_data_federated_stackoverflow_lr
from fedml_api.data_preprocessing.stackoverflow_nwp.data_loader import load_partition_data_federated_stackoverflow_nwp
from fedml_api.data_preprocessing.ImageNet.data_loader import load_partition_data_ImageNet
from fedml_api.data_preprocessing.Landmarks.data_loader import load_partition_data_landmarks
from fedml_api.model.cv.mobilenet import mobilenet
from fedml_api.model.cv.resnet import resnet56
from fedml_api.model.cv.cnn import CNN_DropOut
from fedml_api.model.cv.cnn import CNN_OriginalFedAvg
from fedml_api.model.cv.cnn import FedSP_CNN_Original
from fedml_api.model.cv.vgg import *
from fedml_api.data_preprocessing.FederatedEMNIST.data_loader import load_partition_data_federated_emnist

from fedml_api.data_preprocessing.MNIST.data_loader import load_partition_data_mnist
from fedml_api.model.linear.lr import LogisticRegression
from fedml_api.model.cv.resnet_gn import resnet18

from fedml_api.standalone.fedprox.fedprox_api import FedProxAPI
from fedml_api.standalone.fedsp.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from fedml_api.standalone.fedsp.fedsp_api import FedSPAPI

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='cnn_ori', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='mnist', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/femnist',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='sgd',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.03, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_in_total', type=int, default=1000, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
                        help='number of workers')

    parser.add_argument('--comm_round', type=int, default=200,
                        help='how many round of communications we should use')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--mu', type=float, default=0,
                        help='the mu of fedprox')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
    return parser

def load_mnist_data(batch_size):
    logging.info("load_data. dataset_name = mnist")
    client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = load_partition_data_mnist(batch_size)
    client_num_in_total = client_num

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset

def load_femnist_data():
    logging.info("load_data. dataset_name = femnist")
    client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = load_partition_data_federated_emnist(args.dataset, args.data_dir)
    args.client_num_in_total = client_num

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def custom_model_trainer(model):
    return MyModelTrainerCLS(model)

if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


    parser = add_args(argparse.ArgumentParser(description='FedProx-standalone'))
    args = parser.parse_args()
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)
    # load data
    wandb.init(
        project="test_cnn",
        name="FedProx-mu_" + str(args.mu) + "r_" + str(args.comm_round) + "-e_" + str(args.epochs) + "-b_" + str(
            args.batch_size) + "-lr_" + str(args.lr) + "-model_" + str(args.model),
        config=args
    )
    # batch_size = 10
    # dataset = load_mnist_data(batch_size)
    dataset = load_femnist_data()

    model = FedSP_CNN_Original(output_dim = dataset[7])

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedprox)
    model_trainer = custom_model_trainer(model)
    logging.info(model)

    fedspAPI = FedSPAPI(dataset, device, args, model_trainer)
    fedspAPI.train()
