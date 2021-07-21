# Script to convert images into tabular data for QMNIST
import pickle
import numpy as np
import cv2 as cv
import tensorflow as tf
from PIL import Image

from tensorflow.keras.models import Model


# Resize the images
def resize_image(images):
    resized_images = []
    for img in images:
        temp_img = cv.resize(img, (32,32), interpolation=4)
        temp_img = Image.fromarray(temp_img.astype('uint8')).convert('RGB')
        final_img = np.array(temp_img)
        resized_images.append(final_img)
    resized_images = np.array(resized_images)
    return resized_images


if __name__ == '__main__':

    # Load QMNIST data from QMNIST.pickle
    pickle_file = './QMNIST_ppml.pickle'
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
        x_defender = pickle_data['x_defender']
        x_reserve = pickle_data['x_reserve']
        y_defender = pickle_data['y_defender']
        y_reserve = pickle_data['y_reserve']
    del pickle_data
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
    with open('QMNIST_tabular_ppml.pickle', 'wb') as f:
        pickle.dump(split_dict, f)

    print('Tabular data stored.')
