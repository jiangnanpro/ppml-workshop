"""
Oracle attack of the unsupervised domain adaptation network: from (large or not) Fake-MNIST to QMNIST defender.
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

from DeepDA_code import *

def load_trained_model(model_path, device):
    if model_path in ["supervised_model_checkpoints/resnet50_fm_defender.pth",
                      "supervised_model_checkpoints/resnet50_large_fm_defender.pth"]:
        model = TransferNet(10, base_net='resnet50', transfer_loss='lmmd', 
            use_bottleneck=True, bottleneck_width=256, max_iter=1000)
        model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model = models.resnet50(pretrained=False)
        fc_in_features = model.fc.in_features
        model.fc = nn.Linear(fc_in_features, 10)
        model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="supervised_model_checkpoints/resnet50_fm_defender.pth", 
        help="""which trained model to load""")
    parser.add_argument("--dataset_path", type=str, default="data/QMNIST_ppml.pickle", 
        help="""which trained model to load""")
    parser.add_argument("--N", type=int, default=3000, 
        help="""Only the first N samples of defender and reserve data will be used, 
        this means 2 * N samples in total.""")
    parser.add_argument("--attack_mode", type=str, default="forward_target_domain", 
        help="""how to do the one-step attack to the unsupervised domain adaptation model""")
    parser.add_argument("--lr", type=float, default=1e-3, 
        help="""step size of the one-step gradient update, also known as eta""")
    parser.add_argument("--momentum", type=float, default=0.9, 
        help="""SGD momentum of the one-step gradient update""")
    parser.add_argument("--weight_decay", type=float, default=5e-4, 
        help="""weight decay of the one-step gradient update""")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--zip", action="store_true", default=False)
    parser.add_argument("--source_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    return args


def get_transform(device):
    """
    Input: numpy.ndarray, shape = (28, 28), uint8
    Output: torch.Tensor (cpu or cuda), torch.Size([1, 3, 224, 224]), torch.float32
    """
    transform = torchvision.transforms.Compose(
        [lambda x: Image.fromarray(x),
         lambda x: x.convert("RGB") if x.mode == "L" else x,
         torchvision.transforms.Resize([224, 224]),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]),
         lambda x: torch.unsqueeze(x, 0),
         lambda x: x.to(device)],
    )
    return transform


def convert_to_one_hot_encoding(y):
    one_hot = np.zeros(10, dtype=np.int64)
    one_hot[y] = 1
    return one_hot


def form_x_all_y_all(args):
    """
    x_all:
        numpy.ndarray, shape = (2 * N, 28, 28), uint8
    y_all:
        numpy.ndarray (2 * N, 10) int64

    First N are defender data, the last N are reserve data
    """
    if args.dataset_path == "data/QMNIST_ppml.pickle":
        with open(args.dataset_path, 'rb') as f:
            pickle_data = pickle.load(f)
            x_defender = pickle_data['x_defender']
            x_reserve = pickle_data['x_reserve']
            y_defender = pickle_data['y_defender']
            y_reserve = pickle_data['y_reserve']
    else: 
        with open('data/QMNIST_ppml.pickle', 'rb') as f:
            pickle_data = pickle.load(f)
            x_defender = pickle_data['x_defender']
            x_reserve = pickle_data['x_reserve']

        with open("data/y_defender_flipped20.pickle", "rb") as f:
            y_defender = pickle.load(f).astype(int)
            y_defender = y_defender.argmax(axis=1).reshape((-1, 1))

        with open("data/y_reserve_flipped20.pickle", "rb") as f:
            y_reserve = pickle.load(f).astype(int)
            y_reserve = y_reserve.argmax(axis=1).reshape((-1, 1))

    x_all = []
    y_all = []

    for i in range(args.N):
        x_all.append(x_defender[i])
        y_all.append(convert_to_one_hot_encoding(y_defender[i, 0]))

    for i in range(args.N):
        x_all.append(x_reserve[i])
        y_all.append(convert_to_one_hot_encoding(y_reserve[i, 0]))

    x_all = np.array(x_all)
    y_all = np.array(y_all)

    return x_all, y_all


def save_np_array(results_dir, file_name, arr):
    file_path = os.path.join(results_dir, file_name)
    with open(file_path, "wb") as f:
        np.save(f, arr)
    print("{} saved.".format(file_path)) 


def evaluate_model(model, data, transform, args):
    """
    res: predicted probabilities
    """
    model.eval()
    res = []
    for i in range(data.shape[0]):
        img = transform(data[i])
        if args.model_path in ["supervised_model_checkpoints/resnet50_fm_defender.pth", 
        "supervised_model_checkpoints/resnet50_large_fm_defender.pth"]:
            res.append(F.softmax(model.predict(img).squeeze(0), dim=0).detach().to("cpu").numpy())
        else:
            res.append(F.softmax(model(img).squeeze(0), dim=0).detach().to("cpu").numpy())
        
    res = np.array(res)
    return res


def compute_yhat_all(args, device, x_all, transform, results_dir):
    """
    yhat_all:
        numpy.ndarray, shape = (2 * N, 10), float32
    """
    model = load_trained_model(args.model_path, device)
    yhat_all = evaluate_model(model, x_all, transform, args)
    save_np_array(results_dir, "yhat_all.npy", yhat_all)
    return yhat_all


def compute_total_gradient_norm(model):
    total_gradient_norm = 0
    for p in model.parameters():
        param_gradient_norm = p.grad.detach().data.norm(2)
        total_gradient_norm += param_gradient_norm.item() ** 2
    total_gradient_norm = total_gradient_norm ** 0.5
    return total_gradient_norm


def compute_perturbed_model(args, device, transform, img, label, iter_source=None):
    """
    returns the model updated by one extra gradient step
    and returns the norm of the gradient
    """
    model = load_trained_model(args.model_path, device)

    if args.model_path in ["supervised_model_checkpoints/resnet50_fm_defender.pth", 
    "supervised_model_checkpoints/resnet50_large_fm_defender.pth"]:
        params = model.get_parameters(initial_lr=args.lr)
    else:
        params = model.parameters()
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, 
        weight_decay=args.weight_decay, nesterov=False)

    model.train()
    
    if args.attack_mode == "forward_target_domain":
        if args.model_path in ["supervised_model_checkpoints/resnet50_fm_defender.pth", 
        "supervised_model_checkpoints/resnet50_large_fm_defender.pth"]:
            output = model.predict(img) 
        else:
            output = model(img) 
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        total_gradient_norm = compute_total_gradient_norm(model)
        optimizer.step()
    elif args.attack_mode == "transfer_loss":
        data_source, label_source = next(iter_source)
        data_source, label_source = data_source.to(device), label_source.to(device)
        clf_loss, transfer_loss = model(data_source, img, label_source)
        clf_loss_weight = 0
        transfer_loss_weight = 0.5
        loss = clf_loss_weight * clf_loss + transfer_loss_weight * transfer_loss
        
        optimizer.zero_grad()
        loss.backward()
        total_gradient_norm = compute_total_gradient_norm(model)
        optimizer.step()
    elif args.attack_mode == "total_loss":
        data_source, label_source = next(iter_source)
        data_source, label_source = data_source.to(device), label_source.to(device)
        clf_loss, transfer_loss = model(data_source, img, label_source)
        clf_loss_weight = 1
        transfer_loss_weight = 0.5
        loss = clf_loss_weight * clf_loss + transfer_loss_weight * transfer_loss

        optimizer.zero_grad()
        loss.backward()
        total_gradient_norm = compute_total_gradient_norm(model)
        optimizer.step()
    else:
        raise NotImplementedError("attack_mode={} not supported.".format(args.attack_mode))
    return model, total_gradient_norm


def compute_oneStep_yhat_all_gradNorm_all(args, device, x_all, y_all, transform, results_dir):
    gradNorm_all = []
    oneStep_yhat_all = []


    if args.attack_mode in ["transfer_loss", "total_loss"]:
        # for both resnet50_fm_defender.pth and resnet50_large_fm_defender.pth, 
        # seed was 52 in the beginning
        set_random_seed(52)
        source_dataloader = load_source_dataloader(args.source_path, 32, num_workers=args.num_workers)
        iter_source = iter(source_dataloader)
    else:
        iter_source = None

    for idx in range(x_all.shape[0]):
        # img: torch.Tensor, torch.Size([1, 3, 224, 224]), torch.float32
        img = transform(x_all[idx]) 

        # label: torch.Tensor, shape = (1,), int64
        label = torch.from_numpy(np.expand_dims(np.argmax(y_all[idx]), axis=0)).to(device)

        model, total_gradient_norm = compute_perturbed_model(args, device, transform, img, label, 
            iter_source)
        gradNorm_all.append(total_gradient_norm)
        oneStep_yhat = evaluate_model(model, 
            np.expand_dims(x_all[idx], axis=0), transform, args).squeeze(0)
        oneStep_yhat_all.append(oneStep_yhat)

    oneStep_yhat_all = np.array(oneStep_yhat_all)
    gradNorm_all = np.array(gradNorm_all)

    save_np_array(results_dir, "oneStep_yhat_all.npy", oneStep_yhat_all)
    save_np_array(results_dir, "gradNorm_all.npy", gradNorm_all)

    return oneStep_yhat_all, gradNorm_all


if __name__ == "__main__":

    t0 = time.time()

    args = parse_arguments()

    results_dir = args.results_dir # "results_oracle_attack_UDA"
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)


    torch.backends.cudnn.deterministic = True
    random.seed(2021)
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    np.random.seed(2021)

    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Using CPU for PyTorch")
    else:
        device = torch.device("cuda")
        print("Using GPU for PyTorch")

    

    x_all, y_all = form_x_all_y_all(args)

    transform = get_transform(device)

    yhat_all = compute_yhat_all(args, device, x_all, transform, results_dir)

    oneStep_yhat_all, gradNorm_all = compute_oneStep_yhat_all_gradNorm_all(args, 
        device, x_all, y_all, transform, results_dir)
    

    if args.zip:
        cmd = "zip -r {}.zip {}".format(results_dir, results_dir)
        subprocess.call(cmd.split())

    print("Done in {:.1f} s.".format(time.time() - t0))









