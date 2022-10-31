# part b
import os

import tensorflow as tf
import pandas as pd
import shutil
IMAGE_SIZE = (224, 224)
IMAGE_PATH = "./Project_A_Supp/mhist_dataset/images"
ANNOTATION_PATH = "./Project_A_Supp/mhist_dataset/annotations.csv"
IMAGE_PATH_WITH_LABEL = "./Project_A_Supp/mhist_dataset/images_with_label"


def file_generator():
    annotation = pd.read_csv(ANNOTATION_PATH)
    for index, line in annotation.iterrows():
        original = "./Project_A_Supp/mhist_dataset/images/{}".format(line["Image Name"])
        target = "./Project_A_Supp/mhist_dataset/images_with_label/{}/{}/". \
            format(line["Partition"], line["Number of Annotators who Selected SSA (Out of 7)"])
        try:
            shutil.copy(original, target)
        except IOError as io_err:
            os.makedirs(os.path.dirname(target))
            shutil.copy(original, target)


def data_loader():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "./Project_A_Supp/mhist_dataset/images_with_label/train",
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(224, 224),
        batch_size=32)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        "./Project_A_Supp/mhist_dataset/images_with_label/train",
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(224, 224),
        batch_size=32)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        "./Project_A_Supp/mhist_dataset/images_with_label/test",
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        seed=123,
        image_size=(224, 224),
        batch_size=32)
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    file_generator()
    train_ds, val_ds, test_ds = data_loader()
