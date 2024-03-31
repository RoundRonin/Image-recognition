

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import History

from keras.callbacks import ReduceLROnPlateau
from tensorflow import data as tf_data

class model:

    model: keras.Model
    history: History
    callbacks = []

    def __init__(self, image_size: tuple[int,int], num_classes: int, dense_layers_number: int, conv_descriptor: tuple):
        
        model = Sequential()

        model.add(keras.Input(shape=image_size + (1, )))

        model.add(Conv2D(16, (3,3), 1, activation='relu'))
        model.add(MaxPooling2D())
        # model.add(Dropout(0.25))

        model.add(Conv2D(32, (3,3), 1, activation='relu'))
        model.add(MaxPooling2D())
        # model.add(Dropout(0.25))

        model.add(Conv2D(16, (3,3), 1, activation='relu'))
        model.add(MaxPooling2D())
        # model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        # model.add(Dropout(0.25))

        model.add(Dense(num_classes, activation = "softmax"))

        self.model = model

        model.summary()

    def compile(self):

        # Компиляция модели
        optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08)
        self.model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
        # self.model.summary()

    def train(self, train_ds: tf_data.Dataset, epochs: int, validation_data: tf_data.Dataset):
        
        self.history = self.model.fit(train_ds, epochs = epochs, validation_data = validation_data, callbacks=self.callbacks)
    
    def init_learning_rate_reduction(self):
        
        learning_rate_reduction = ReduceLROnPlateau(
            monitor='accuracy',
            patience=3, 
            verbose=1, 
            factor=0.5, 
            min_lr=0.00001
        )

        self.callbacks.append(learning_rate_reduction)
    
    def init_save_at_epoch(self):
        
        self.callbacks.append(keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"))