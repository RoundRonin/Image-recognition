

import clang
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from keras.callbacks import History
import keras
from scikitplot.metrics import plot_confusion_matrix 
from tensorflow import data as tf_data


class plotter_evaluator:

    model: keras.Model
    history: History
    class_names: list
    labels: np.ndarray
    pred_labels: list

    def __init__(self, history: History, model: keras.Model, class_names: set):
        self.history = history
        self.model = model

        if type(list(class_names)[0]) is not str:
            self.class_names=list(map(str,class_names))
        else:
            self.class_names = list(class_names)

    def calc_predictions(self, test_values: tf_data.Dataset):

        labels =  np.array([])
        for _, y in test_values: # type: ignore
            labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

        self.labels = labels

        Predictions = self.model.predict(test_values)
        self.pred_labels = np.argmax(Predictions, axis = 1)

    def plot_loss_accuracy(self):
        
        history = self.history
        fig = plt.figure(figsize=(9,5))

        plt.subplot(211)
        plt.plot(history.history['loss'], color='teal', label='loss')
        plt.plot(history.history['val_loss'], color='orange', label='val_loss')
        plt.legend(loc="upper left")

        plt.subplot(212)
        plt.plot(history.history['accuracy'], color='teal', label='accuracy')
        plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
        plt.legend(loc="upper left")

        fig.suptitle('Loss and Accuracy', fontsize=19)
        plt.show()

    def print_report(self):

        print(classification_report(self.labels, self.pred_labels, target_names=self.class_names))

    def plot_confusion_matrix(self):
        
        plot_confusion_matrix(self.labels, self.pred_labels,cmap= 'YlGnBu')
        plt.show()


import cv2
import tensorflow as tf
import math


class tester:

    img: np.ndarray | None = None
    model: keras.Model
    path: str

    def __init__(self, model: keras.Model):
        self.model = model

    def read_image(self, path: str, show: bool):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if(img is None):
            print("Image not found")
            return

        if(show): plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        self.path = path
        self.img = img

    def img_predict(self):
        if(self.img is None):
            print("No image supplied")
            return

        prediction = self.model.predict(np.expand_dims(self.img/255, 0))
        probabilities = list(map(lambda x: math.floor(x*1000)/1000, prediction[0]))
        print(self.path)
        print(probabilities)

