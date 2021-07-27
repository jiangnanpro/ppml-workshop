# Script to reproduce the split of QMNIST.pickle
import pickle
import os

from torchvision.datasets import CIFAR10 # version 0.9.1
import numpy as np # version 1.19.5

if __name__=='__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Load QMNIST data from torchvision.datasets
    cifar_train = CIFAR10('cifar10', download=True, train=True)
    cifar_test = CIFAR10('cifar10', download=False, train=False)
    
    cifar_labels_train = np.array(cifar_train.targets).reshape([-1,1])
    cifar_labels_test = np.array(cifar_test.targets).reshape([-1,1])

    cifar_images = np.vstack((cifar_train.data, cifar_test.data))
    cifar_labels = np.vstack((cifar_labels_train, cifar_labels_test))

    cifar_len = cifar_images.shape[0]

    # Set the size of each partition of the split
    defender_partition_size = int(cifar_len/2)
    reserve_partition_size = cifar_len-defender_partition_size

    # Randomly selecting the images for each partition
    rng = np.random.RandomState(2021)
    random_seq = rng.choice(cifar_len,size=cifar_len, replace=False)

    defender_partition_indexes = random_seq[0:defender_partition_size]
    reserve_partition_indexes = random_seq[defender_partition_size:]

    # Create dict of partitions
    split_dict = dict()
    split_dict['x_defender'] = cifar_images[defender_partition_indexes]
    split_dict['x_reserve'] = cifar_images[reserve_partition_indexes]
    split_dict['y_defender'] = cifar_labels[defender_partition_indexes]
    split_dict['y_reserve'] = cifar_labels[reserve_partition_indexes]

    # Store the dict using pickle
    with open(os.path.join(current_dir,'CIFAR10_ppml.pickle'), 'wb') as f:
        pickle.dump(split_dict, f)