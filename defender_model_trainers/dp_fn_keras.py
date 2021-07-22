#!/usr/bin/env python
# coding: utf-8
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import tensorflow as tf
import random as python_random
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer

from data.utils import load_qmnist_data

np.random.seed(20)
python_random.seed(123)
tf.random.set_seed(3)

NUM_CLASSES = 10

def defender_model_fn_dp():
    """The architecture of the defender (victim) model.
    The attack is white-box, hence the attacker is assumed to know this architecture too."""

    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax)
    ])
        
    train_op = DPKerasAdamOptimizer(l2_norm_clip=1.0,
        noise_multiplier=1.1,
        num_microbatches=1,
        learning_rate=1e-4
       )
    
    model.compile(optimizer=train_op,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__=='__main__':

    # Load the data
    pickle_file = os.path.join(parent_dir,'data/QMNIST_tabular_ppml.pickle')
    x_defender, _, y_defender, _ = load_qmnist_data(pickle_file)
    print('Data loaded.')
    y_defender = tf.keras.utils.to_categorical(y_defender[:,0],num_classes=NUM_CLASSES)

    # Experiments: train the model several times varying the training set size
    n_partitions = [48, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 200000]

    for n_examples in n_partitions:
        model = defender_model_fn_dp()

        with tf.device("cpu:0"):
            model.fit(x_defender[:n_examples], y_defender[:n_examples], epochs=100, batch_size=64, validation_split=0.5, shuffle=False)

        model.save(os.path.join(parent_dir,'defender_trained_models','dp_fn_{}examples'.format(n_examples)))