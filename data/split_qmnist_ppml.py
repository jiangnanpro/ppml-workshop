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
defender_partition_ratio = 1/5
reserved_partition_ratio = 4/5

defender_partition_size = int(qmnist_len*defender_partition_ratio)+10
reserved_partition_size = qmnist_len-defender_partition_size

# Randomly selecting the images for each partition
rng = np.random.RandomState(2021)
random_seq = rng.choice(qmnist_len,size=qmnist_len, replace=False)

defender_partition_indexes = random_seq[0:defender_partition_size]
reserved_partition_indexes = random_seq[defender_partition_size:]

# Create dict of partitions
split_dict = dict()
split_dict['x_defender'] = qmnist_images[defender_partition_indexes]
split_dict['x_reserve'] = qmnist_images[reserved_partition_indexes]
split_dict['y_defender'] = qmnist_targets[defender_partition_indexes]
split_dict['y_reserve'] = qmnist_targets[reserved_partition_indexes]

# Store the dict using pickle
with open('QMNIST_ppml.pickle', 'wb') as f:
    pickle.dump(split_dict, f)