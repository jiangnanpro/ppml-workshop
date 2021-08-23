import pickle

import numpy as np
import cv2 as cv
from PIL import Image  

def load_data(pickle_file):    
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
        x_defender = pickle_data['x_defender']
        x_reserve = pickle_data['x_reserve']
        y_defender = pickle_data['y_defender']
        y_reserve = pickle_data['y_reserve']
    return x_defender, x_reserve, y_defender, y_reserve

def resize_image(images):
    resized_images = []
    for img in images:
        temp_img = cv.resize(img, (32,32), interpolation=4)
        temp_img = Image.fromarray(temp_img.astype('uint8')).convert('RGB')
        final_img = np.array(temp_img)
        resized_images.append(final_img)
    return np.array(resized_images)
