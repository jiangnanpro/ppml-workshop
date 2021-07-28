"""
Input: QMNIST_ppml.pickle
Output: The same content of QMNIST_ppml.pickle but in torchvision.datasets.ImageFolder format

torchvision.datasets.ImageFolder is a format that can be easily consumed 
by various algorithms (incl. domain adaptation) implemented in PyTorch

See:
* https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.ImageFolder


Note that in addition to the usual torchvision.datasets.ImageFolder format, we also 
saved a file raw_labels.csv which stores the 8-column labels of QMNIST. See: 
* https://github.com/facebookresearch/qmnist#21-using-the-qmnist-extended-testing-set


To be used with https://github.com/SunHaozhe/transferlearning/tree/save_checkpoint
"""


import os
import glob
import time
import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
import shutil
import pickle
from PIL import Image # 8.0.1


def to_ImageFolder_format(X, y, parent_dir, dataset_name):
    """
    https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.ImageFolder
    
    Note that in addition to the usual torchvision.datasets.ImageFolder format, we also 
    saved a file raw_labels.csv which stores the 8-column labels of QMNIST. See: 
    * https://github.com/facebookresearch/qmnist#21-using-the-qmnist-extended-testing-set
    """
    dir_ = os.path.join(parent_dir, dataset_name)
    if os.path.exists(dir_):
        shutil.rmtree(dir_)

    for digit in range(10):
        os.makedirs(get_digit_path(dir_, digit))

    df = []
    for idx, img in enumerate(X):
        img = Image.fromarray(img, mode="L")
        label = y[idx, 0] # Character class: 0 to 9
        image_name = "{}_{}.png".format(dataset_name, idx)
        image_path = os.path.join(get_digit_path(dir_, label), image_name)
        img.save(image_path)
        df.append((image_name, image_path, *y[idx]))

    # https://github.com/facebookresearch/qmnist#22-using-the-qmnist-extended-labels
    df = pd.DataFrame(df, columns=["image_name", "image_path", "character_class", 
        "nist_hsf_series", "nist_writer_id", "digit_index_for_this_writer", 
        "nist_class_code", "global_nist_digit_index", "duplicate", "unused"])

    df.to_csv(os.path.join(dir_, "raw_labels.csv"), encoding="utf-8")

    




def get_digit_path(dir_, digit):
    return os.path.join(dir_, str(digit))



if __name__ == "__main__":

    with open('QMNIST_ppml.pickle', 'rb') as f:
        pickle_data = pickle.load(f)
        x_defender = pickle_data['x_defender']
        x_reserve = pickle_data['x_reserve']
        y_defender = pickle_data['y_defender']
        y_reserve = pickle_data['y_reserve']


    parent_dir = "QMNIST_ppml_ImageFolder"
    if os.path.exists(parent_dir):
        shutil.rmtree(parent_dir)


    print(x_defender.shape)
    print(x_reserve.shape)
    print(y_defender.shape)
    print(y_reserve.shape)

    t0 = time.time()
    to_ImageFolder_format(x_defender, y_defender, parent_dir, "defender")
    print("Defender formatting, done in {:.1f} s.".format(time.time() - t0))

    t0 = time.time()
    to_ImageFolder_format(x_reserve, y_reserve, parent_dir, "reserve")
    print("Reserve formatting, done in {:.1f} s.".format(time.time() - t0))

    











