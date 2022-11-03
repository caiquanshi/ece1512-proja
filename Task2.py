# part b
import os
import tensorflow as tf
import pandas as pd
import shutil
import numpy
import matplotlib.pyplot as plt

from keras_flops import get_flops
#####################################
# Title: keras_flops
# Author: tokusumi
# Date: 17 Aug 2020
# Availability: https://github.com/tokusumi/keras-flops
#
#####################################



IMAGE_SIZE = (224, 224)
IMAGE_PATH = "./Project_A_Supp/mhist_dataset/images/"
ANNOTATION_PATH = "./Project_A_Supp/mhist_dataset/annotations.csv"
IMAGE_PATH_WITH_VOTE = "./images_with_label_vote/"
IMAGE_PATH_WITH_MAJORITY = "./images_with_label_majority/"
IMAGE_SHAPE = (224, 224, 3)
NUM_CLASSES = 2
BATCH_SIZE = 16


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
        image_dir+"train",
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        image_dir+"train",
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        image_dir+"test",
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE)
    return train_ds, val_ds, test_ds


def create_pretrain_resnet_model():
    res_net_model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet',
                                                               input_shape=IMAGE_SHAPE)

    for i in range(len(res_net_model.layers)):
        res_net_model.layers[i].trainable = False


    classifier = tf.keras.layers.Flatten()(res_net_model.output)

    classifier = tf.keras.layers.Dense(NUM_CLASSES)(classifier)
    re_model = tf.keras.models.Model(inputs=res_net_model.input, outputs=classifier)
    # re_model.summary()
    return re_model


def create_mobilenet_model():
    mobile_net_model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet',
                                                                      input_shape=IMAGE_SHAPE)
    for i in range(len(mobile_net_model.layers)):
        mobile_net_model.layers[i].trainable = False

    classifier = tf.keras.layers.Flatten()(mobile_net_model.output)

    classifier = tf.keras.layers.Dense(NUM_CLASSES)(classifier)
    mb_model = tf.keras.models.Model(inputs=mobile_net_model.input, outputs=classifier)
    # mb_model.summary()
    return mb_model
def parameter_count(model):
  trainableParams = numpy.sum([numpy.prod(v.get_shape()) for v in model.trainable_weights])
  nonTrainableParams = numpy.sum([numpy.prod(v.get_shape()) for v in model.non_trainable_weights])
  return trainableParams,nonTrainableParams

def compute_num_correct(model, images, labels):
  """Compute number of correctly classified images in a batch.

  Args:
    model: Instance of tf.keras.Model.
    images: Tensor representing a batch of images.
    labels: Tensor representing a batch of labels.

  Returns:
    Number of correctly classified images.
  """
  class_logits = model(images, training=True)
  return tf.reduce_sum(
      tf.cast(tf.math.equal(tf.argmax(class_logits, -1), tf.argmax(labels, -1)),
              tf.float32)), tf.argmax(class_logits, -1), tf.argmax(labels, -1)

def compute_F1(model, images, labels):
  """Compute number of F1 scores in a batch.

  Args:
    model: Instance of tf.keras.Model.
    images: Tensor representing a batch of images.
    labels: Tensor representing a batch of labels.

  Returns:
    array of four F1 numbers.
  """
  TP=0
  FP=0
  FN=0
  TN=0
  class_logits = model(images, training=True)
  class_logits =tf.argmax(class_logits,-1)
  labels=tf.argmax(labels,-1)
  for i in range(tf.size(labels).numpy()):
      if class_logits[i]==0 and labels[i]==0:
          TP+=1
      elif class_logits[i]==1 and labels[i]==1:
          TN+=1
      elif class_logits[i]==1 and labels[i]==0:
          FN+=1
      elif class_logits[i]==0 and labels[i]==1:
          FP+=1
  return [TN,TP,FP,FN]


def train_and_eveluate_transfer_learn(model, train_data, test_data, initial_num_epochs, fine_tune_num_epochs,
                                      ft, compute_loss, lr):

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr)
    test_acc = []
    test_f1 = []
    for epoch in range(1, initial_num_epochs + 1):
        # Run initial training.
        print('initial Epoch {}: '.format(epoch), end='')
        for images, labels in train_data:
            with tf.GradientTape() as tape:
                # your code start from here for step 4
                loss_value = compute_loss(model, images, labels)

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        F1_list = [0, 0, 0, 0]
        for images, labels in test_data:
            # your code start from here for step 4
            F1_list = numpy.add(F1_list, compute_F1(model, images, labels))
        prec_1 = F1_list[1] / (F1_list[1] + F1_list[2])
        recall_1 = F1_list[1] / (F1_list[1] + F1_list[3])
        F1_1 = 2 * prec_1 * recall_1 / (prec_1 + recall_1)
        prec_2 = F1_list[0] / (F1_list[0] + F1_list[3])
        recall_2 = F1_list[0] / (F1_list[0] + F1_list[2])
        F1_2 = 2 * prec_2 * recall_2 / (prec_2 + recall_2)
        test_f1.append((F1_1, F1_2))
        print("F1 score:", round(F1_1, 4), round(F1_2, 4))

        num_correct = 0
        num_total = 977
        for images, labels in test_data:
            # your code start from here for step 4

            num_correct += compute_num_correct(model, images, labels)[0]
        print("Class_accuracy: " + '{:.2f}%'.format(
            num_correct / num_total * 100))
        test_acc.append(num_correct / num_total)

    for i in range(len(model.layers) - 3, len(model.layers) - ft - 3, -1):
        model.layers[i].trainable = True
    for i in range(len(model.layers)-1, len(model.layers) - 3, -1):
        model.layers[i].trainable = False
    # model.summary()
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr*0.1)
    for fine_tune_epoch in range(1, fine_tune_num_epochs + 1):
        # Run fine tune training.
        print('Fine_tune_Epoch {}: '.format(fine_tune_epoch), end='')
        for images, labels in train_data:
            with tf.GradientTape() as tape:
                # your code start from here for step 4
                loss_value = compute_loss(model, images, labels)

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        F1_list = [0, 0, 0, 0]
        for images, labels in test_data:
            # your code start from here for step 4
            F1_list = numpy.add(F1_list, compute_F1(model, images, labels))
        prec_1 = F1_list[1] / (F1_list[1] + F1_list[2])
        recall_1 = F1_list[1] / (F1_list[1] + F1_list[3])
        F1_1 = 2 * prec_1 * recall_1 / (prec_1 + recall_1)
        prec_2 = F1_list[0] / (F1_list[0] + F1_list[3])
        recall_2 = F1_list[0] / (F1_list[0] + F1_list[2])
        F1_2 = 2 * prec_2 * recall_2 / (prec_2 + recall_2)
        test_f1.append((F1_1, F1_2))
        print("F1 score:", round(F1_1, 4), round(F1_2, 4))

        # Run evaluation.
        num_correct = 0
        num_total = 977
        for images, labels in test_data:
            # your code start from here for step 4

            num_correct += compute_num_correct(model, images, labels)[0]
        print("Class_accuracy: " + '{:.2f}%'.format(
            num_correct / num_total * 100))
        test_acc.append(num_correct / num_total)

    return test_acc, test_f1


def train_and_evaluate_distillation(teacher_model, student_model, train_data, test_data, initial_num_epochs,
                                    fine_tune_num_epochs, ft, aplha, temperature, lr):

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr)
    test_acc = []
    test_f1 = []
    for epoch in range(1, initial_num_epochs + 1):
        # Run initial training.
        print('initial Epoch {}: '.format(epoch), end='')
        for images, labels in train_data:
            with tf.GradientTape() as tape:
                # your code start from here for step 4
                loss_value = compute_student_loss(student_model, teacher_model, images, labels, temperature, aplha)

            grads = tape.gradient(loss_value, student_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, student_model.trainable_variables))

        F1_list = [0, 0, 0, 0]
        for images, labels in test_data:
            # your code start from here for step 4
            F1_list = numpy.add(F1_list, compute_F1(student_model, images, labels))
        prec_1 = F1_list[1] / (F1_list[1] + F1_list[2])
        recall_1 = F1_list[1] / (F1_list[1] + F1_list[3])
        F1_1 = 2 * prec_1 * recall_1 / (prec_1 + recall_1)
        prec_2 = F1_list[0] / (F1_list[0] + F1_list[3])
        recall_2 = F1_list[0] / (F1_list[0] + F1_list[2])
        F1_2 = 2 * prec_2 * recall_2 / (prec_2 + recall_2)
        test_f1.append((F1_1, F1_2))
        print("F1 score:", round(F1_1, 4), round(F1_2, 4))
        num_correct = 0
        num_total = 977
        for images, labels in test_data:

            num_correct += compute_num_correct(student_model, images, labels)[0]
        print("Class_accuracy: " + '{:.2f}%'.format(
            num_correct / num_total * 100))
        test_acc.append(num_correct / num_total)

    for i in range(len(student_model.layers) - 3, len(student_model.layers) - ft - 3, -1):
        student_model.layers[i].trainable = True
    for i in range(len(student_model.layers) - 1, len(student_model.layers) - 3, -1):
        student_model.layers[i].trainable = False
    # student_model.summary()
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr*0.1)
    for fine_tune_epoch in range(1, fine_tune_num_epochs + 1):
        # Run fine tune training.
        print('Fine_tune_Epoch {}: '.format(fine_tune_epoch), end='')
        for images, labels in train_data:
            with tf.GradientTape() as tape:
                # your code start from here for step 4
                loss_value = compute_student_loss(student_model, teacher_model, images, labels, temperature, aplha)

            grads = tape.gradient(loss_value, student_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, student_model.trainable_variables))

        F1_list = [0, 0, 0, 0]
        for images, labels in test_data:
            # your code start from here for step 4
            F1_list = numpy.add(F1_list, compute_F1(student_model, images, labels))
        prec_1 = F1_list[1] / (F1_list[1] + F1_list[2])
        recall_1 = F1_list[1] / (F1_list[1] + F1_list[3])
        F1_1 = 2 * prec_1 * recall_1 / (prec_1 + recall_1)
        prec_2 = F1_list[0] / (F1_list[0] + F1_list[3])
        recall_2 = F1_list[0] / (F1_list[0] + F1_list[2])
        F1_2 = 2 * prec_2 * recall_2 / (prec_2 + recall_2)
        test_f1.append((F1_1, F1_2))
        print("F1 score:", round(F1_1, 4), round(F1_2, 4))

        # Run evaluation.
        num_correct = 0
        num_total = 977
        for images, labels in test_data:
            # your code start from here for step 4

            num_correct += compute_num_correct(student_model, images, labels)[0]
        print("Class_accuracy: " + '{:.2f}%'.format(
            num_correct / num_total * 100))
        test_acc.append(num_correct / num_total)

    return test_acc, test_f1


def distillation_loss(teacher_logits: tf.Tensor, student_logits: tf.Tensor,
                      temperature):

  soft_targets = tf.exp(teacher_logits/temperature) / tf.\
      reduce_sum(tf.exp(teacher_logits/temperature), -1, keepdims=True)

  return tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(
          soft_targets, student_logits / temperature)) * temperature ** 2

def compute_student_loss(student_model, teacher_model, images, labels, temperature, alpha):


  student_subclass_logits = student_model(images, training=True)

  teacher_subclass_logits = teacher_model(images, training=False)


  # Compute subclass distillation loss between student subclass logits and
  # softened teacher subclass targets probabilities.



  distillation_loss_value = distillation_loss(teacher_subclass_logits, student_subclass_logits, temperature)

  # Compute cross-entropy loss with hard targets.



  cross_entropy_loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                                                    logits=teacher_subclass_logits))

  return alpha*distillation_loss_value+(1-alpha)*cross_entropy_loss_value

def compute_loss_fun(model, images, labels):

  subclass_logits = model(images, training=True)

  cross_entropy_loss_value=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=subclass_logits))

  return cross_entropy_loss_value

def plot_graph(title, test_acc, test_f1):
    eps = range(1, 36)
    plt.title(title)
    plt.plot(eps, test_acc, label='test accuracy')
    plt.plot(eps, [i[0] for i in test_f1], label='f1 score for HP')
    plt.plot(eps, [i[1] for i in test_f1], label='f1 score for SSA')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(title +".png")
    plt.show()
    plt.clf()

if __name__ == "__main__":
    # tf.debugging.set_log_device_placement(True)
    # file_generator_by_vote(IMAGE_PATH, IMAGE_PATH_WITH_VOTE)
    file_generator_by_majority(IMAGE_PATH, IMAGE_PATH_WITH_MAJORITY)
    train_ds, val_ds, test_ds = data_loader(IMAGE_PATH_WITH_MAJORITY)
    print("transfer learn resnet start here")
    res_net = create_pretrain_resnet_model()
    test_acc, test_f1 = train_and_eveluate_transfer_learn(res_net, train_ds, test_ds, 10, 25, 10, compute_loss_fun, 1e-4)
    plot_graph("Teacher_model(resnet)_figure", test_acc, test_f1)
    print("distillation start here")
    mobile_net = create_mobilenet_model()
    test_acc, test_f1 = train_and_evaluate_distillation(res_net, mobile_net, train_ds, test_ds, 10, 25, 10, 0.5, 4, 1e-3)
    plot_graph("Student_model(mobilenet)_distillation_figure", test_acc, test_f1)
    print("train mobilnet start here")
    mobile_net = create_mobilenet_model()
    test_acc, test_f1 = train_and_eveluate_transfer_learn(mobile_net, train_ds, test_ds, 10, 25, 10, compute_loss_fun, 1e-3)
    plot_graph("Student_model(mobilenet)_from_scratch_figure", test_acc, test_f1)
    print("teacher trainable parameters & non-trainable parameters:", parameter_count(res_net))
    print("student trainable parameters & non-trainable parameters:", parameter_count(mobile_net))
    teacher_flops = get_flops(res_net, batch_size=1)
    student_flops = get_flops(mobile_net, batch_size=1)
    print(f"teacher FLOPS: {teacher_flops / 10 ** 9:.03} G")
    print(f"student FLOPS: {student_flops / 10 ** 9:.03} G")
