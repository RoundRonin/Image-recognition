

import clang
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import keras.callbacks as cb
import keras as k
from scikitplot.metrics import plot_confusion_matrix 

class plotter_evaluator:

    model: k.Model
    history: cb.History
    class_names: list
    labels: np.ndarray
    pred_labels: list

    def __init__(self, history: cb.History, model: k.Model, class_names: set):
        self.history = history
        self.model = model

        if type(list(class_names)[0]) is not str:
            self.class_names=list(map(str,class_names))
        else:
            self.class_names = list(class_names)

    def calc_predictions(self, test_values: list):

        labels =  np.array([])
        for _, y in test_values:
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
        
        fig, ax = plt.subplots(1, 2, figsize = (25,  8))
        ax = plot_confusion_matrix(self.labels, self.pred_labels, ax = ax[0], cmap= 'YlGnBu')