import numpy as np
import tensorflow as tf

import ModelTraining as mt


class ImageClassifier(object):
    def __init__(self, path_to_model):
        self.model = tf.keras.experimental.load_from_saved_model(path_to_model)

    def get_classification(self, img):
        resize_img = img.resize((mt.resize_width, mt.resize_height))
        img_expanded = np.expand_dims(np.array(resize_img, dtype=np.uint8).mean(2), axis=0)
        return np.argmax(self.model.predict(img_expanded)[0])
