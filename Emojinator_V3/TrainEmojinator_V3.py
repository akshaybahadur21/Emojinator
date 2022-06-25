from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

image_x, image_y = 200, 200
batch_size = 64
train_dir = "/Users/3482704/Akshay_git/Emojinator/Emojinator_V3/gestures"


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
    filepath = "emojinator_v3.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    return model, callbacks_list


def main():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        rotation_range=15,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
        fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_x, image_y),
        color_mode="grayscale",
        batch_size=batch_size,
        seed=42,
        class_mode='categorical',
        subset="training")

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_x, image_y),
        color_mode="grayscale",
        batch_size=batch_size,
        seed=42,
        class_mode='categorical',
        subset="validation")

    model, callbacks_list = keras_model(image_x, image_y)
    model.fit_generator(train_generator, epochs=5, validation_data=validation_generator)
    scores = model.evaluate_generator(generator=validation_generator, steps=64)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

    model.save('emojinator_v3.h5')


main()
