"""CLI interface for image_recognition project.

Be creative! do whatever you want!

- Install click or typer and create a CLI app
- Use builtin argparse
- Start a web application
- Import things from your .base module
"""
import image_recognition.base as base
from image_recognition.visualization import plotter_evaluator

import numpy as np

import os
import keras
from keras import layers
from tensorflow import data as tf_data

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import RMSprop

from keras.callbacks import ReduceLROnPlateau


def main():  # pragma: no cover

    base.hello()
    ## Формирование классов на основе файловой структуры

    # Внутри указанной директории должны нахдиться папки, имя которых соотвествует классу.
    # В папках -- изображения, соответсвующие классу.

    # Размер по вертикали, размер по горизонтали. К этим значениям будут приведены все изображения (сжаты/растянуты, не обрезаны)
    height = 140
    width = 90
    image_size = (height, width)

    # Больше -- быстрее, меньше -- точнее. В теории.
    batch_size = 32

    # Вышеописанная директория
    path_to_data = "Data50"

    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        path_to_data,
        validation_split=0.2,
        subset="both",
        seed=1337,
        label_mode="categorical",
        shuffle=True,
        image_size=image_size,
        batch_size=batch_size,
        color_mode="grayscale",
        crop_to_aspect_ratio=True,
    )
    
    # Получение имён классов, числа классов.
    # TODO: Rework
    labels =  np.array([])
    for x, y in val_ds:
        labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

    class_names = set(labels)
    num_classes = len(class_names)

    print(len(labels))

    ## Обработка данных

    # Вносится рандомизация (ротация, зум, перемещение). Также приводится яркость к понятному нейросети формату (вместо 0-255, 0-1).

    data_augmentation_layers = [
        layers.RandomRotation(0.08),
        layers.RandomZoom(
            height_factor=[-0.2,0.2],
            width_factor=[-0.2,0.2],
            fill_mode="constant",
            fill_value=255.0,
        ),
        layers.RandomTranslation(
            height_factor = [-0.1, 0.1],
            width_factor = [-0.1, 0.1],
            fill_mode="constant",
            fill_value=255.0,
        ),
        layers.Rescaling(1.0 / 255)
    ]

    def data_augmentation(images):
        for layer in data_augmentation_layers:
            images = layer(images)
        return images

    ### Применение слоёв обработки данных

    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf_data.AUTOTUNE,
    )

    ## Формирование модели

    # Модель последовательная. Состоит из слоёв, каждый из которых исполняется после предыдущего.
    # В первом слое описывается форма подаваемых данных. Первые два параметра -- размеры изображения (описаны в начале).
    # Третий пораметр: 1 -- ч/б изображение, 2 -- RGB, 3 -- RGBA

    # Последний слой имеет число нейронов, равное количеству классов.

    model = Sequential()

    model.add(keras.Input(shape=(height,width,1)))

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

    # Компиляция модели
    optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08)
    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

    model.summary()

    # Обучение нейросети

    # Число проходов по набору данных. Не всегда улучшает результат. Надо смотреть на графики. (50 по умолчанию, при малом наборе данных)
    epochs = 50

    learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy',
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
    callbacks = [
        # keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
        learning_rate_reduction
    ]

    history = model.fit(train_ds, epochs = epochs, validation_data = val_ds, callbacks=callbacks)

    # Визуализация

    pe = plotter_evaluator(history, model, class_names)
    pe.calc_predictions(val_ds)

    ## Графики потерь и точности

    # Высокой должна быть и accuracy и val_accuracy. Первая -- точность на обучающей выборке, вторая -- на тестовой. 
    # Когда/если точность на обучающей выборке начинает превосходить точность на тестовой, продолжать обучение не следует.

    # Потери (loss) должны быть низкими.

    pe.plot_loss_accuracy()

    ## Вычисление отчёта о качестве классификации

    # Значения accuracy, recall, f1 должны быть высокими.

    pe.print_report()

    ## Матрица запутанности

    # Хорший способ понять, как именно нейросеть ошибается

    pe.plot_confusion_matrix()
