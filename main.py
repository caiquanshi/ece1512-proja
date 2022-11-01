# part b
import os
from tensorflow.python.client import device_lib
import tensorflow as tf
import pandas as pd
import shutil
IMAGE_SIZE = (224, 224)
IMAGE_PATH = "./Project_A_Supp/mhist_dataset/images/"
ANNOTATION_PATH = "./Project_A_Supp/mhist_dataset/annotations.csv"
IMAGE_PATH_WITH_VOTE = "./images_with_label_vote"
IMAGE_PATH_WITH_MAJORITY = "./images_with_label_majority"
IMAGE_SHAPE = (224, 224, 3)
NUM_CLASSES = 2


def file_generator_by_vote(mhist_dir, new_dir, annotation_path=ANNOTATION_PATH):
    annotation = pd.read_csv(annotation_path)
    for index, line in annotation.iterrows():
        original = mhist_dir + "{}".format(line["Image Name"])
        target = new_dir + "{}/{}/". \
            format(line["Partition"], line["Number of Annotators who Selected SSA (Out of 7)"])
        try:
            shutil.copy(original, target)
        except IOError as io_err:
            os.makedirs(os.path.dirname(target))
            shutil.copy(original, target)


def file_generator_by_majority(mhist_dir, new_dir, annotation_path=ANNOTATION_PATH):
    annotation = pd.read_csv(annotation_path)
    for index, line in annotation.iterrows():
        original = mhist_dir + "{}".format(line["Image Name"])
        target = new_dir + "{}/{}/". \
            format(line["Partition"], line["Majority Vote Label"])
        try:
            shutil.copy(original, target)
        except IOError as io_err:
            os.makedirs(os.path.dirname(target))
            shutil.copy(original, target)


def data_loader(image_dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        image_dir+"/train",
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(224, 224),
        batch_size=32)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        image_dir+"/train",
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(224, 224),
        batch_size=32)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        image_dir+"/test",
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        seed=123,
        image_size=(224, 224),
        batch_size=32)
    return train_ds, val_ds, test_ds


def create_pretrain_resnet_model(ft=0):
    res_net_model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet',
                                                               input_shape=IMAGE_SHAPE)

    for i in range(len(res_net_model.layers)-ft):
        res_net_model.layers[i].trainable = False

    # classifier = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(res_net_model.output)
    # classifier = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(classifier)
    classifier = tf.keras.layers.Flatten()(res_net_model.output)
    # classifier = tf.keras.layers.Dense(100, activation='relu')(classifier)
    classifier = tf.keras.layers.Dense(NUM_CLASSES)(classifier)
    re_model = tf.keras.models.Model(inputs=res_net_model.input, outputs=classifier)
    re_model.summary()
    return re_model

def compute_num_correct(model, images, labels):
  """Compute number of correctly classified images in a batch.

  Args:
    model: Instance of tf.keras.Model.
    images: Tensor representing a batch of images.
    labels: Tensor representing a batch of labels.

  Returns:
    Number of correctly classified images.
  """
  class_logits = model(images, training=False)
  return tf.reduce_sum(
      tf.cast(tf.math.equal(tf.argmax(class_logits, -1), tf.argmax(labels, -1)),
              tf.float32)), tf.argmax(class_logits, -1), tf.argmax(labels, -1)


def train_and_eveluate(model, train_data, test_data, num_epochs, compute_loss):
    # your code start from here for step 4
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001)
    test_acc = []
    for epoch in range(1, num_epochs + 1):
        # Run training.
        print('Epoch {}: '.format(epoch), end='')
        for images, labels in train_data:
            with tf.GradientTape() as tape:
                # your code start from here for step 4
                loss_value = compute_loss(model, images, labels)

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Run evaluation.
        num_correct = 0

        num_total = 977
        for images, labels in test_data:
            # your code start from here for step 4

            num_correct += compute_num_correct(model, images, labels)[0]
        print("Class_accuracy: " + '{:.2f}%'.format(
            num_correct / num_total * 100))
        test_acc.append(num_correct / num_total * 100)

    return max(test_acc)


def compute_loss_fun(model, images, labels):
  """Compute subclass knowledge distillation teacher loss for given images
     and labels.

  Args:
    images: Tensor representing a batch of images.
    labels: Tensor representing a batch of labels.

  Returns:
    Scalar loss Tensor.
  """
  subclass_logits = model(images, training=True)

  cross_entropy_loss_value=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=subclass_logits))

  return cross_entropy_loss_value


if __name__ == "__main__":
    # tf.debugging.set_log_device_placement(True)
    # file_generator_by_vote(IMAGE_PATH, IMAGE_PATH_WITH_LABEL)
    file_generator_by_majority(IMAGE_PATH, IMAGE_PATH_WITH_MAJORITY)
    train_ds, val_ds, test_ds = data_loader(IMAGE_PATH_WITH_MAJORITY)
    res_net = create_pretrain_resnet_model(10)
    train_and_eveluate(res_net, train_ds, test_ds, 12, compute_loss_fun)