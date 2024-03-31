

import keras
from tensorflow import data as tf_data
import numpy as np
import os

class importer:

    __SEED = 42
    __data_augmentation_layers: list

    train_ds: tf_data.Dataset
    validation_ds: tf_data.Dataset

    class_names: list[str]
    num_classes: int

    def __init__(self, image_size: tuple[int, int], batch_size: int, data_directory: str, validation_split: float | None = None):
       
        dirlist = os.listdir(data_directory)
        subset = None
        if (validation_split is not None): subset = "both"

        data = keras.utils.image_dataset_from_directory(
            data_directory,
            validation_split=validation_split,
            subset=subset,
            class_names=dirlist,
            seed=self.__SEED,
            label_mode="categorical",
            shuffle=True,
            image_size=image_size,
            batch_size=batch_size,
            color_mode="grayscale",
            crop_to_aspect_ratio=True,
        )

        if (validation_split is not None): self.train_ds, self.validation_ds = data
        else: self.train_ds = data #type: ignore

        self.class_names = dirlist
        self.num_classes = len(dirlist)

    def generate_augmentation_layers(self, zoom_factor: float, move_factor: float, rotation_factor: float):
        
        self.__data_augmentation_layers = [
            keras.layers.RandomRotation(rotation_factor),
            keras.layers.RandomZoom(
                height_factor=[-zoom_factor,zoom_factor],
                width_factor=[-zoom_factor,zoom_factor],
                fill_mode="constant",
                fill_value=255.0
            ),
            keras.layers.RandomTranslation(
                height_factor = [-move_factor, move_factor],
                width_factor = [-move_factor, move_factor],
                fill_mode="constant",
                fill_value=255.0,
            ),
            keras.layers.Rescaling(1.0 / 255)
        ]
    
    def apply_augmentation(self):
        
        # TODO Rework to duplicate and augment data
        self.train_ds = self.train_ds.map(
            lambda img, label: (self.__data_augmentation(img), label),
            num_parallel_calls=tf_data.AUTOTUNE,
        )

    def __data_augmentation(self, images):

        for layer in self.__data_augmentation_layers:
            images = layer(images)
        return images
