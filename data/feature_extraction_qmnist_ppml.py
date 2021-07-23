# Script to convert images into tabular data for QMNIST
import pickle
import os

import tensorflow as tf
from tensorflow.keras.models import Model

from utils import load_qmnist_data, resize_image


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Load QMNIST data from QMNIST.pickle
    pickle_file = os.path.join(current_dir, 'QMNIST_ppml.pickle')
    x_defender, x_reserve, y_defender, y_reserve = load_qmnist_data(pickle_file)
    print('Data loaded.')

    x_defender = resize_image(x_defender)
    x_reserve = resize_image(x_reserve)

    base_model = tf.keras.applications.VGG19(weights='imagenet',include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

    x_defender_features = model.predict(x_defender)
    x_reserve_features = model.predict(x_reserve)

    x_defender_tabular = x_defender_features
    x_reserve_tabular = x_reserve_features

    x_defender_tabular = x_defender_tabular.squeeze()
    x_reserve_tabular = x_reserve_tabular.squeeze()
    print(x_defender_tabular.shape)
    # Create dict of partitions

    split_dict = dict()
    split_dict['x_defender'] = x_defender_tabular
    split_dict['x_reserve'] = x_reserve_tabular
    split_dict['y_defender'] = y_defender
    split_dict['y_reserve'] = y_reserve

    # Store the dict using pickle
    with open(os.path.join(current_dir, 'QMNIST_tabular_ppml.pickle'), 'wb') as f:
        pickle.dump(split_dict, f)

    print('Tabular data stored.')
