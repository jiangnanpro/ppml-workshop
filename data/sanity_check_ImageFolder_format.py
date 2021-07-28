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
import argparse
from PIL import Image # 8.0.1





if __name__ == "__main__":

    t0 = time.time()

    for dataset_name in ["defender", "reserve"]:
        dir_ = os.path.join("QMNIST_ppml_ImageFolder", dataset_name)

        indices_to_plot = list(range(9))

        if not os.path.exists(dir_):
            print("Directory {} not found.".format(dir_))
        else:
            items = []
            for path in glob.glob(os.path.join(dir_, "**", "*.png"), recursive=True):
                basename = os.path.basename(path)
                index = int(os.path.splitext(basename)[0].split("_")[-1])
                if index in indices_to_plot:
                    items.append((index, path))

        items = sorted(items, key=lambda x: x[0])

        fig, ax = plt.subplots(3,3, figsize = (10,10))
        n=0
        for i in range(3):
            for j in range(3):
                img = Image.open(items[n][1]).convert("L")
                ax[i,j].imshow(img)
                ax[i,j].set_title(items[n][1].split(os.sep)[2])
                n=n+1
        fig.savefig("sanity_check_{}.png".format(dataset_name), dpi=fig.dpi)


    print("Done in {:.1f} s.".format(time.time() - t0))











