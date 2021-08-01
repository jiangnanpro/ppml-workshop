
import os
import sys
import glob
import datetime


import argparse
import collections
import itertools
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim
from torchvision import transforms, models, datasets


import wandb

from DeepDA_code import *



def val(val_dataloader, model, device):
    correct_cnt = 0
    total_cnt = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(val_dataloader):
            X, y = X.to(device), y.to(device)

            logits = model.predict(X)
            
            correct_cnt += (logits.argmax(dim=1) == y).sum().item() 
            total_cnt += y.shape[0]

    val_acc = (correct_cnt / total_cnt) * 100

    return val_acc





def get_dataloaders(args):
    transform = {
        'train': transforms.Compose(
            [lambda x: x.convert("RGB") if x.mode == "L" else x,
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]),
        'test': transforms.Compose(
            [lambda x: x.convert("RGB") if x.mode == "L" else x,
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    }

    train_dataset = datasets.ImageFolder(root=args.train_path, transform=transform["train"])
    val_dataset = datasets.ImageFolder(root=args.val_path, transform=transform["test"])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_dataloader, val_dataloader

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


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train_path', type=str, default="data/QMNIST_ppml_ImageFolder/defender")
    argparser.add_argument('--val_path', type=str, default="data/QMNIST_ppml_ImageFolder/reserve")
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--lr', type=float, help='learning rate of the optimizer', default=1e-3)
    argparser.add_argument('--momentum', type=float, default=0.9)
    argparser.add_argument('--weight_decay', type=float, default=1e-4)
    argparser.add_argument('--scheduler_patience', type=int, default=5)
    argparser.add_argument('--scheduler_factor', type=float, default=0.1)
    argparser.add_argument('--epochs', type=int, help='how many epochs in total', default=30)
    argparser.add_argument('--random_seed', type=int, help='random seed', default=68)
    argparser.add_argument('--train_mode', type=str, default="whole")
    argparser.add_argument('--random_labels', action="store_true", default=False)
    argparser.add_argument('--overfit', action="store_true", default=False)
    argparser.add_argument('--num_workers', type=int, default=0)
    args = argparser.parse_args()

    if args.random_labels:
        assert args.train_path != "data/QMNIST_ppml_ImageFolder/defender"
        assert args.val_path != "data/QMNIST_ppml_ImageFolder/reserve"


    t0_overall = time.time()

    # random seed
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)

    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU for PyTorch")
    else:
        device = torch.device('cpu')
        print("Using CPU for PyTorch")


    # neural network

    ## "supervised_model_checkpoints/resnet50_large_fm_defender.pth"
    model_path = "supervised_model_checkpoints/resnet50_fm_defender.pth" 
    model = TransferNet(10, base_net='resnet50', transfer_loss='lmmd', 
            use_bottleneck=True, bottleneck_width=256, max_iter=1000)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    print("Model created.")

    if args.random_labels:
        labels_status = "flipped"
    else:
        labels_status = "normal"

    if args.overfit:
        try_overfitting = "overfit"
    else:
        try_overfitting = "normal"
    
    # wandb
    project_name = "tmpFODFS"
    group_name = "tmp"
    #group_name = "{}-{}-{}-{}".format(args.train_mode, args.weight_decay, labels_status, try_overfitting)
    wandb_dir = "wandb_logs"
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)
    wandb.init(config=args, project=project_name, dir=wandb_dir)
    
    env_info = get_torch_gpu_environment()
    for k, v in env_info.items():
        wandb.run.summary[k] = v
    wandb.run.summary["trainable_parameters_count"] = count_trainable_parameters(model)
    wandb_run_name = wandb.run.name


    # optimizer and scheduler
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 
                                                           "min", 
                                                           verbose=True, 
                                                           patience=args.scheduler_patience, 
                                                           factor=args.scheduler_factor)
    print("Optimizer and scheduler ready.")

    # data loaders
    train_dataloader, val_dataloader = get_dataloaders(args)

    


    # checkpoint setting
    checkpoints_dir = "supervised_model_checkpoints"
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    best_model = [np.inf, os.path.join(checkpoints_dir, 
        "best_model_{}_{}_{}.pth".format(project_name, group_name, wandb_run_name))] # (score, path)
    

    last_model = os.path.join(checkpoints_dir, "last_model_{}_{}_{}.pth".format(project_name, group_name, wandb_run_name))

    val_acc = val(val_dataloader, model, device)

    print()
    print()

    print("val_acc = ")
    print(val_acc)

    print()
    print()

    
    t_overall = time.time() - t0_overall 
    print("Done in {:.2f} s.".format(t_overall))
    wandb.run.summary["overall_computation_time"] = t_overall




