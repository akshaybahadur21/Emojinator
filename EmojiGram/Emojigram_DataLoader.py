from __future__ import division
import cv2
import os
import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt
from itertools import islice

LIMIT = None

data_dictionary = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6' : 6, '7': 7,
                   '8': 8, '9': 9, '10': 10, '11': 11,
                   '12': 12}


GESTURE_FOLDER = 'Emojigram_data/'


def return_data():
    X = []
    y = []
    features = []

    for filename in os.listdir(os.path.join(GESTURE_FOLDER)):
        for file in os.listdir(os.path.join(GESTURE_FOLDER, filename)):
            if file.endswith('.jpg'):
                full_path = os.path.join(GESTURE_FOLDER, filename, file)
                X.append(full_path)
                y.append(data_dictionary[filename])

    for i in range(len(X)):
        img = plt.imread(X[i])
        features.append(img)

    features = np.array(features).astype('float32')
    labels = np.array(y).astype('float32')

    with open("features", "wb") as f:
        pickle.dump(features, f, protocol=4)
    with open("labels", "wb") as f:
        pickle.dump(labels, f, protocol=4)


return_data()