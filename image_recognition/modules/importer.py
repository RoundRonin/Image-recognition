

import keras
from tensorflow import data as tf_data
import numpy as np

class importer:

    __SEED = 42
    __data_augmentation_layers: list

    train_ds: tf_data.Dataset
    validation_ds: tf_data.Dataset

    class_names: set
    num_classes: int

    def __init__(self, image_size, batch_size, data_directory, validation_split):
       
        self.train_ds, self.validation_ds = keras.utils.image_dataset_from_directory(
            data_directory,
            validation_split=validation_split,
            subset="both",
            seed=self.__SEED,
            label_mode="categorical",
            shuffle=True,
            image_size=image_size,
            batch_size=batch_size,
            color_mode="grayscale",
            crop_to_aspect_ratio=True,
        )

        self.__get_stats()

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
        
        self.train_ds = self.train_ds.map(
            lambda img, label: (self.__data_augmentation(img), label),
            num_parallel_calls=tf_data.AUTOTUNE,
        )

    def __data_augmentation(self, images):
        for layer in self.__data_augmentation_layers:
            images = layer(images)
        return images

    def __get_stats(self):

        # TODO: Rework
        labels = np.array([])
        for _, y in self.validation_ds: # type: ignore
            labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

        self.class_names = set(labels)
        self.num_classes = len(self.class_names)

