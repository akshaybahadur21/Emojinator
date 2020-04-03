import pickle

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Sequential
from keras.utils import np_utils, print_summary
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def keras_model(image_x, image_y):
    num_of_classes = 13
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_of_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "emojigram.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    callbacks_list.append(TensorBoard(log_dir='Emojigram_logs'))

    return model, callbacks_list


def loadFromPickle():
    with open("features", "rb") as f:
        features = np.array(pickle.load(f))
    with open("labels", "rb") as f:
        labels = np.array(pickle.load(f))

    return features, labels


def loadData():
    data = pd.read_csv("emojigram_data.csv")
    dataset = np.array(data)
    np.random.shuffle(dataset)
    features = dataset[:, 1:2501]
    features = features / 255.
    labels = dataset[:, 0]
    labels = labels.reshape(labels.shape[0], 1)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                        test_size=0.2)
    return train_x, test_x, train_y, test_y


def reshapeData(train_x, test_x, train_y, test_y):
    train_y = np_utils.to_categorical(train_y)
    test_y = np_utils.to_categorical(test_y)
    train_x = train_x.reshape(train_x.shape[0], 50, 50, 1)
    test_x = test_x.reshape(test_x.shape[0], 50, 50, 1)
    return train_x, test_x, train_y, test_y


def printInfo(train_x, test_x, train_y, test_y):
    print("number of training examples = " + str(train_x.shape[0]))
    print("number of test examples = " + str(test_x.shape[0]))
    print("X_train shape: " + str(train_x.shape))
    print("Y_train shape: " + str(train_y.shape))
    print("X_test shape: " + str(test_x.shape))
    print("Y_test shape: " + str(test_y.shape))


def main():
    features, labels = loadFromPickle()
    features = features / 127.5 - 1.
    features, labels = shuffle(features, labels)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                        test_size=0.3)
    train_x, test_x, train_y, test_y = reshapeData(train_x, test_x, train_y, test_y)
    printInfo(train_x, test_x, train_y, test_y)
    model, callbacks_list = keras_model(train_x.shape[1], train_x.shape[2])
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=1, batch_size=64,
              callbacks=callbacks_list)
    scores = model.evaluate(test_x, test_y, verbose=1)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
    print_summary(model)

    model.save('emojigram.h5')


if __name__ == '__main__':
    main()
