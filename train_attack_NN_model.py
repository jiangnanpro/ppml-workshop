"""

"""

import os
import glob
import time
import random
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F  
import torchvision
import matplotlib.pyplot as plt
import shutil
import pickle
from PIL import Image # 8.0.1
import argparse
import subprocess
import wandb

from DeepDA_code import *

"""
To limit the usage of RAM and computation

Justification:
The significance of gradient (as well as activation) computations 
for a membership inference attack varies over the layers of a deep neural 
network. The first layers tend to contain less information about the specific 
data points in the training set, compared to non-member data record from 
the same underlying distribution.

Nasr, Milad, Reza Shokri, and Amir Houmansadr. “Comprehensive Privacy Analysis of 
Deep Learning: Passive and Active White-Box Inference Attacks against Centralized 
and Federated Learning.” 2019 IEEE Symposium on Security and Privacy (SP), May 
2019, 739–53. https://doi.org/10.1109/SP.2019.00065.
"""
selected_conv_layer_names = ["layer4.2.conv3", "layer4.2.conv2", "layer3.5.conv2", 
                             "layer2.3.conv2", "layer1.0.conv1"]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=3000, 
        help="""Only the first N samples of defender and reserve data will be used, 
        this means 2 * N samples in total.""")
    parser.add_argument("--input_feature_path", type=str, 
        default="supervised_normal_whole_attack_using_NN", 
        help="""where to load the input feature of the white-box attacker neural network.""")
    parser.add_argument("--save_attacker_model_path", type=str, 
        default="attacker_NN_model_checkpoints", 
        help="""where to save the checkpoints of the white-box attacker neural network.""")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--random_seed', type=int, help='random seed', default=68)
    parser.add_argument('--lr', type=float, help='learning rate of the optimizer', default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--zip", action="store_true", default=False)
    args = parser.parse_args()
    return args


def load_input_features(args):
    """
    L_all: 
        list of python scalars

    hidden_all: 
        list of dict, each dict: str ==> 3D float32 np.array

    gradients_all: 
        list of dict, each dict: str ==> 4D float32 np.array

    y_all: 
        numpy.ndarray (2 * N, 10), float32 (one-hot encoding)

    yhat_all: 
        numpy.ndarray, shape = (2 * N, 10), float32

    input_features:
        list of InputFeature objects, 2 * N in total
    """
    with open(os.path.join(args.input_feature_path, "L_all.pkl"), "rb") as f:
        L_all = pickle.load(f) 

    with open(os.path.join(args.input_feature_path, "hidden_all.pkl"), "rb") as f:
        hidden_all = pickle.load(f) 

    with open(os.path.join(args.input_feature_path, "gradients_all.pkl"), "rb") as f:
        gradients_all = pickle.load(f) 

    with open(os.path.join(args.input_feature_path, "y_all.npy"), "rb") as f:
        y_all = np.load(f)

    with open(os.path.join(args.input_feature_path, "yhat_all.npy"), "rb") as f:
        yhat_all = np.load(f)

    input_features = group_input_features(args, L_all, hidden_all, gradients_all, y_all, yhat_all)

    return input_features


def group_input_features(args, L_all, hidden_all, gradients_all, y_all, yhat_all):
    """
    concatenate L_all, y_all, yhat_all to one flatten vector
    to be fed into a fully-connected feature extractor
    """
    assert len(L_all) == args.N * 2
    assert len(hidden_all) == args.N * 2
    assert len(gradients_all) == args.N * 2
    assert len(y_all) == args.N * 2
    assert len(yhat_all) == args.N * 2

    input_features = []

    for i in range(args.N * 2):
        L = L_all[i]
        hidden = hidden_all[i]
        grad = gradients_all[i]
        y = y_all[i]
        yhat = yhat_all[i]

        input_feature = InputFeature(L, hidden, grad, y, yhat)
        input_features.append(input_feature)
    return input_features


class InputFeature:
    """
    L: python float32 scalar
    hidden: dict from str to 3D float32 np.array
    grad: dict from str to 4D float32 np.array
    y: np.ndarray of size 10, float32 (one-hot encoding)
    yhat: np.ndarray of size 10, float32 
    """
    def __init__(self, L, hidden, grad, y, yhat):
        self.flatten_vector = [L]
        self.flatten_vector.extend(y.tolist())
        self.flatten_vector.extend(yhat.tolist())
        self.flatten_vector = np.array(self.flatten_vector).astype(np.float32)

        #self.keys = []
        #for k in grad.keys():
        #    self.keys.append(k)

        self.grad = grad
        self.hidden = hidden


class FullyConnectedFeatureExtractor(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__() 
        self.fc1 = nn.Linear(dim_in, 10)
        self.fc2 = nn.Linear(10, dim_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class FullyConnectedEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__() 
        self.fc1 = nn.Linear(dim_in, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, dim_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class Convolutional2DFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = torch.flatten(x)
        return x

class Convolutional3DFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__() 
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = torch.flatten(x)
        return x

def get_feature_shapes(feature_shapes_dir):
    with open(os.path.join(feature_shapes_dir, "hidden_layer_features_shape.pkl"), "rb") as f:
        hidden_layer_features_shape = pickle.load(f)
    with open(os.path.join(feature_shapes_dir, "param_gradients_shape.pkl"), "rb") as f:
        param_gradients_shape = pickle.load(f)
    return hidden_layer_features_shape, param_gradients_shape


class WhiteBoxAttackerNeuralNetwork(nn.Module):
    def __init__(self, selected_conv_layer_names, hidden_layer_features_shape, param_gradients_shape, device):
        super().__init__() 

        self.selected_conv_layer_names = selected_conv_layer_names

        self.hidden_feature_extractors = {}
        for name in self.selected_conv_layer_names:
            channels = hidden_layer_features_shape[name][0]
            self.hidden_feature_extractors[name] = Convolutional2DFeatureExtractor(channels, 5)

        self.grad_feature_extractors = {}
        for name in self.selected_conv_layer_names:
            channels = param_gradients_shape[name][0]
            self.grad_feature_extractors[name] = Convolutional3DFeatureExtractor(channels, 5)


        self.fc_feature_extractor = FullyConnectedFeatureExtractor(21, 5)

        # find it out empirically
        encoder_input_size = 10 # TODO

        self.encoder = FullyConnectedEncoder(encoder_input_size, 1)

        self.device = device


    def forward(self, input_feature):
        x_list = [self.fc_feature_extractor(input_feature.flatten_vector.to(self.device))]

        hidden_embedding = []
        grad_embedding = []
        for name in self.selected_conv_layer_names:
            x_list.append(self.hidden_feature_extractors[name](input_feature.hidden.to(self.device)))
            x_list.append(self.grad_feature_extractors[name](input_feature.grad.to(self.device)))

        x = torch.cat(x_list, dim=0).view(-1)
        x = self.encoder(x)
        return x


def create_attacker_model(device):
    hidden_layer_features_shape, param_gradients_shape = get_feature_shapes(
        feature_shapes_dir="attack_with_NN_resnet50_feature_shapes")

    attacker_model = WhiteBoxAttackerNeuralNetwork(selected_conv_layer_names, 
        hidden_layer_features_shape, param_gradients_shape, device)
    attacker_model.to(device)
    return attacker_model


def count_trainable_parameters(model):
    return sum([x.numel() for x in model.parameters() if x.requires_grad])


def get_torch_gpu_environment():
    env_info = dict()
    env_info["PyTorch_version"] = torch.__version__

    if torch.cuda.is_available():
        env_info["cuda_version"] = torch.version.cuda
        env_info["cuDNN_version"] = torch.backends.cudnn.version()
        env_info["nb_available_GPUs"] = torch.cuda.device_count()
        env_info["current_GPU_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        env_info["nb_available_GPUs"] = 0
    return env_info


def train(train_dataloader, model, device, optim, epoch, args):
    t0 = time.time()

    train_loss = 0
    correct_cnt = 0
    total_cnt = 0

    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for batch_idx, (X, y) in enumerate(train_dataloader):
        optim.zero_grad()

        logits = model(X)
        loss = criterion(logits, y)

        loss.backward()
        optim.step()

        train_loss += loss.item()
        correct_cnt += (logits.argmax(dim=1) == y).sum().item() 
        total_cnt += y.shape[0]

    train_loss /= len(train_dataloader)
    train_acc = (correct_cnt / total_cnt) * 100
    
    t1 = time.time() - t0
    print("Epoch {} | Train loss {:.2f} | Train acc {:.2f} | Time {:.1f} seconds.".format(
            epoch+1, train_loss, train_acc, t1))
    wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc, "train_epoch_time": t1})


def test(test_dataloader, model, device, epoch, args, test_logits):
    t0 = time.time()
    
    test_loss = 0
    correct_cnt = 0
    total_cnt = 0

    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_dataloader):
            
            logits = model(X)

            test_logits.extend(logits.detach().cpu().numpy())
            
            loss = criterion(logits, y)

            test_loss += loss.item()
            correct_cnt += (logits.argmax(dim=1) == y).sum().item() 
            total_cnt += y.shape[0]

    
    test_loss /= len(test_dataloader)
    test_acc = (correct_cnt / total_cnt) * 100

    
    t1 = time.time() - t0
    print("Epoch {} | Test loss {:.2f} | Test acc {:.2f} | Time {:.1f} seconds.".format(
            epoch+1, test_loss, test_acc, t1))
    wandb.log({"epoch": epoch+1, "test_loss": test_loss, "test_acc": test_acc, "test_epoch_time": t1})
    wandb.run.summary["final_test_loss"] = test_loss
    wandb.run.summary["final_test_acc"] = test_acc



class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        super().__init__()
        assert len(features) == len(labels)
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]



def get_data_loaders(args, input_features):
    # First N == defender, last N == reserve
    assert len(input_features) == args.N * 2


    defender_train = input_features[: int(args.N // 2)]
    defender_test = input_features[int(args.N // 2) : args.N]
    reserve_train = input_features[args.N : (args.N + int(args.N // 2))]
    reserve_test = input_features[(args.N + int(args.N // 2)) :]

    train_features = defender_train.extend(reserve_train)
    train_labels = np.ones(args.N, dtype=np.int64)
    train_labels[- int(args.N // 2) :] = 0

    test_features = defender_test.extend(reserve_test)
    test_labels = np.ones(args.N, dtype=np.int64)
    test_labels[- int(args.N // 2) :] = 0

    train_dataset = FeatureDataset(train_features, train_labels)
    test_dataset = FeatureDataset(test_features, test_labels)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_dataloader, test_dataloader



if __name__ == "__main__":

    t0 = time.time()

    args = parse_arguments()

    torch.backends.cudnn.deterministic = False
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Using CPU for PyTorch")
    else:
        device = torch.device("cuda")
        print("Using GPU for PyTorch")



    # wandb
    project_name = "white_box_membership_attacker_with_NN"
    group_name = "placeholder".format()
    wandb_dir = "wandb_logs"
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)
    wandb.init(config=args, project=project_name, group=group_name, dir=wandb_dir)
    
    env_info = get_torch_gpu_environment()
    for k, v in env_info.items():
        wandb.run.summary[k] = v
    wandb_run_name = wandb.run.name

    # load model
    model = create_attacker_model(device)

    print("Model ready.")


    wandb.run.summary["trainable_parameters_count"] = count_trainable_parameters(model)


    # load input data
    input_features = load_input_features(args)

    print("input_features loaded.")

    # dataloaders
    train_dataloader, test_dataloader = get_data_loaders(args, input_features)

    print("Dataloaders ready.")

    # optimizer
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    print("Optimizer ready.")

    white_box_attack_NN_results_dir = "white_box_attack_NN_results"
    if not os.path.exists(white_box_attack_NN_results_dir):
        os.makedirs(white_box_attack_NN_results_dir)
    
    test_logits = []

    # train
    print("Training loop starts...")
    for epoch in range(args.epochs):
        train(train_dataloader, model, device, optim, epoch, args)
        test(test_dataloader, model, device, epoch, args, test_logits)

        wandb.log({"epoch": epoch+1, "lr": optim.param_groups[0]['lr']})

    print("Training loop ends.")

    torch.save(model.state_dict(), 
        os.path.join(white_box_attack_NN_results_dir, "resnet50_{}.pth".format(wandb_run_name)))

    # save test_logits with size N (3000)
    ## the true label of the first half is 1 (defender), the true label of the second half is 0 (reserve).
    with open(os.path.join(white_box_attack_NN_results_dir, "test_logits.pkl"), "wb") as f:
        pickle.dump(test_logits, f)


    if args.zip:
        print("Starts zipping the directory {}".format(white_box_attack_NN_results_dir))
        cmd = "zip -r {}.zip {}".format(white_box_attack_NN_results_dir, white_box_attack_NN_results_dir)
        subprocess.call(cmd.split())

    print("Done in {:.1f} s.".format(time.time() - t0))






