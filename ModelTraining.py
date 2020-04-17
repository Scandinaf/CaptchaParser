from __future__ import absolute_import, division, print_function, unicode_literals

import time
from glob import glob

# Вспомогательные библиотеки
import numpy as np
from PIL import Image
from tensorflow import keras

# TensorFlow и tf.keras

resize_width = 23
resize_height = 23
__saved_model_path = "./data/saved_models/{}".format(int(time.time()))


def __load_data(pattern):
    train_images = []
    train_labels = []
    for path in glob(pattern):
        train_images.append(__load_image(path, resize_width, resize_height))
        train_labels.append(__get_label(path))
    return np.array(train_images), np.array(train_labels, dtype=np.uint8)


def __load_image(path, width, height):
    image = Image.open(path).resize((width, height))
    return np.array(image, dtype=np.uint8).mean(2)


def __get_label(path):
    return path.split("/")[-1].split("_")[0]


if __name__ == '__main__':
    (train_images, train_labels) = __load_data("./data/train/*.png")
    (test_images, test_labels) = __load_data("./data/validation/*.png")
    class_names = ['Plus', '1', '2', '3', '4', '5',
                   '6', '7', '8', '9', 'Minus']

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(resize_width, resize_height)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=20)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('\nТочность на проверочных данных:', test_acc)

    keras.experimental.export_saved_model(model, __saved_model_path)
