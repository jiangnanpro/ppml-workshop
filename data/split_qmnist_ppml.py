# Script to reproduce the split of QMNIST.pickle
import pickle

from torchvision.datasets import QMNIST # version 0.9.1
import numpy as np # version 1.19.5

# Load QMNIST data from torchvision.datasets
qall = QMNIST('qmnist', what='nist', download=True, compat=False)

qmnist_images = qall.data.numpy()
qmnist_targets = qall.targets.numpy()

qmnist_len = qmnist_images.shape[0]

# Set the size of each partition of the split
defender_partition_size = 200000
reserve_partition_size = qmnist_len-200000

# Randomly selecting the images for each partition
rng = np.random.RandomState(2021)
random_seq = rng.choice(qmnist_len,size=qmnist_len, replace=False)

defender_partition_indexes = random_seq[0:defender_partition_size]
reserve_partition_indexes = random_seq[defender_partition_size:]

# Create dict of partitions
split_dict = dict()
split_dict['x_defender'] = qmnist_images[defender_partition_indexes]
split_dict['x_reserve'] = qmnist_images[reserve_partition_indexes]
split_dict['y_defender'] = qmnist_targets[defender_partition_indexes]
split_dict['y_reserve'] = qmnist_targets[reserve_partition_indexes]

# Store the dict using pickle
with open('QMNIST_ppml.pickle', 'wb') as f:
    pickle.dump(split_dict, f)