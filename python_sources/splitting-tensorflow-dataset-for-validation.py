#!/usr/bin/env python
# coding: utf-8

# # Splitting TensorFlow Dataset for Validation 
# 
# This notebook shows a way to split a TensorFlow Dataset into two, for training and validation. Only training dataset is provided in this competition. So, to evaluate models, the provided dataset needs to be splitted into two.
# 
# To split dataset, [sklearn.model_selection.train_test_split()](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) is used. This can maintain the ratio of the target value. About 98% is 0 and 1 is less than 2%. By using the result of train_test_split(), the provied TensorFlow dataset is filtered and splitted into two datasets.
# 
# # Table of Contents
# 
# * <a href="#Preparation">Preparation</a>
# * <a href="#Dataset1">Dataset 1</a>
# * <a href="#SplittingTensorflowDataset">Splitting Tensorflow Dataset</a>
# * <a href="#Dataset2">Dataset 2</a>
# * <a href="#Model">Model</a>
# * <a href="#Evaluation">Evaluation</a>
#     * <a href="#LossAucPrecisionRecall">Loss, Auc, Precision, Recall</a>
#     * <a href="#Roc">Roc</a>
#     * <a href="#ConfusionMatrix">Confusion Matrix</a>
# * <a href="#Predictions">Predictions</a>

# # References
# 
# * [Getting started with 100+ flowers on TPU](https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu)
# * [Melanoma TPU EfficientNet B5_dense_head](https://www.kaggle.com/ajaykumar7778/melanoma-tpu-efficientnet-b5-dense-head)
# * [TFRecords 768x768, 512x512, 384x384 With Meta Data](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/155579)
# * [Rotation Augmentation GPU/TPU - [0.96+]](https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96)
# * [Classification on imbalanced data](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)

# <a name="Preparation"></a>
# # Preparation

# In[ ]:


get_ipython().system('pip install -q efficientnet')


# In[ ]:


import numpy as np # linear algebra
import os
print(os.listdir('../input'))


# In[ ]:


import tensorflow as tf

print(tf.__version__)
print(tf.keras.__version__)


# In[ ]:


# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is set.
    # On Kaggle this is always the case.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# For training data, Chris Deotte's 512x512 image with metadata is used.
# Thanks a lot for providing this dataset!
#   - Discussion: [TFRecords 768x768, 512x512, 384x384 With Meta Data](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/155579) 
#   - Dataset: [Melanoma TFRecords 512x512](https://www.kaggle.com/cdeotte/melanoma-512x512/settings)

# In[ ]:


from kaggle_datasets import KaggleDatasets

# you can list the bucket with "!gsutil ls $GCS_DS_PATH"
GCS_DS_PATH = KaggleDatasets().get_gcs_path('melanoma-512x512')


# In[ ]:


EPOCHS = 25
IMAGE_SIZE = [512, 512]
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
GCS_PATH = GCS_DS_PATH

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test*.tfrec')


# <a name="Dataset1"></a>
# # Dataset 1

# In[ ]:


AUTO = tf.data.experimental.AUTOTUNE

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        "patient_id": tf.io.FixedLenFeature([], tf.int64),
        "sex": tf.io.FixedLenFeature([], tf.int64),
        "age_approx": tf.io.FixedLenFeature([], tf.int64),
        "anatom_site_general_challenge": tf.io.FixedLenFeature([], tf.int64),
        "diagnosis": tf.io.FixedLenFeature([], tf.int64),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    image_name = example['image_name']
    target = tf.cast(example['target'], tf.int32)
    return image, target

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        "patient_id": tf.io.FixedLenFeature([], tf.int64),
        "sex": tf.io.FixedLenFeature([], tf.int64),
        "age_approx": tf.io.FixedLenFeature([], tf.int64),
        "anatom_site_general_challenge": tf.io.FixedLenFeature([], tf.int64),
        # no 'diagnosis' and 'target' for the test data.
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    image_name = example['image_name']
    return image, image_name # no target

def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    # automatically interleaves reads from multiple files
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(
        read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    # returns a dataset of (image, target) pairs if labeled=True or
    # (image, image_name) pairs if labeled=False
    return dataset


# <a name="SplittingTensorflowDataset"></a>
# # Splitting Tensorflow Dataset

# In[ ]:


# At first, make a vanilla training dataset by just loading from TFRecord files.
# The order should be maintained.
vanilla_training_ds = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=True)

print(vanilla_training_ds)


# In[ ]:


def print_target_counts(y, label):
    _, y_counts = np.unique(y, return_counts=True)
    y_total = len(y)
    y_0_count = y_counts[0]
    y_1_count = y_counts[1]
    y_1_percent = y_1_count / y_total * 100.0
    print("{0:10s}: Total={1:5d}, 0={2:5d}, 1={3:3d}, ratio of 1={4:.2f}%".format(
        label, y_total, y_0_count, y_1_count, y_1_percent))


# In[ ]:


# Extract target values from the vanilla training dataset.
# Indices are generated along with the target values, which are used to filter dataset.
y_targets = np.array([ target.numpy() for _, target in iter(vanilla_training_ds) ])
X_indices = np.arange(len(y_targets))

print_target_counts(y_targets, "Total")


# In[ ]:


from sklearn.model_selection import train_test_split

# Split the generated indices and target values by train_test_split().
# The ratio of target values should be kept in the splitted datasets.
X_train_indices, X_val_indices, y_train_targets, y_val_targets = train_test_split(
    X_indices, y_targets, test_size=0.1, stratify=y_targets, random_state=53)

print_target_counts(y_train_targets, "Training")
print_target_counts(y_val_targets, "Validation")


# In[ ]:


def get_selected_dataset(ds, X_indices_np):
    # Make a tensor of type tf.int64 to match the one by Dataset.enumerate(). 
    X_indices_ts = tf.constant(X_indices_np, dtype=tf.int64)
    
    def is_index_in(index, rest):
        # Returns True if the specified index value is included in X_indices_ts.
        #
        # '==' compares the specified index value with each values in X_indices_ts.
        # The result is a boolean tensor, looks like [ False, True, ..., False ].
        # reduce_any() returns Ture if True is included in the specified tensor.
        return tf.math.reduce_any(index == X_indices_ts)
    
    def drop_index(index, rest):
        return rest

    # Dataset.enumerate() is similter to Python's enumerate().
    # The method adds indices to each elements. Then, the elements are filtered
    # by using the specified indices. Finally unnecessary indices are dropped.
    selected_ds = ds         .enumerate()         .filter(is_index_in)         .map(drop_index)
    return selected_ds


# In[ ]:


splitted_train_ds = get_selected_dataset(vanilla_training_ds, X_train_indices)
splitted_val_ds = get_selected_dataset(vanilla_training_ds, X_val_indices)

print(splitted_train_ds)
print(splitted_val_ds)


# <a name="Dataset2"></a>
# # Dataset 2

# In[ ]:


def data_augment(image, target):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image, target

def get_training_dataset():
    dataset = splitted_train_ds
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset():
    dataset = splitted_val_ds
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset


# In[ ]:


NUM_TRAINING_IMAGES = len(y_train_targets)
NUM_VALIDATION_IMAGES = len(y_val_targets)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print('Dataset: {} training images, {} validation images'.format(
    NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES))


# <a name="Model"></a>
# # Model
# 
# * [AUC](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC), [Precision](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Precision), and [Recall](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Recall) are used as metrics.
# * Initialize the bias for the final Dense layer to reflect the imbalance of target values as shown in [Optional: Set the correct initial bias](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#optional_set_the_correct_initial_bias).

# In[ ]:


# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#optional_set_the_correct_initial_bias

import math
from tensorflow.keras.initializers import Constant

_, (y_target_neg_count, y_target_pos_count) = np.unique(y_targets, return_counts=True)
y_target_pos_ratio = y_target_pos_count / y_target_neg_count
dence_initial_bias = math.log(y_target_pos_ratio)
dense_bias_initializer = Constant(dence_initial_bias)

print(y_target_neg_count, y_target_pos_count)
print(y_target_pos_ratio)
print(dence_initial_bias)


# In[ ]:


import efficientnet.tfkeras as efn
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import AUC, Precision, Recall

with strategy.scope():
    model = Sequential([
        efn.EfficientNetB0(
            include_top=False, weights='noisy-student',
            input_shape=(*IMAGE_SIZE, 3), pooling='avg'),
        Dropout(
            0.5, name="dropout"),
        Dense(
            1, activation='sigmoid',
            bias_initializer=dense_bias_initializer, name='classify')
    ])
    metrics = [ AUC(name='auc'), Precision(name='precision'), Recall(name='recall') ]    

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
model.summary()


# <a name="Training"></a>
# # Training
# 
# * For learning rate, ramp up and exponential decay is used in [Rotation Augmentation GPU/TPU - [0.96+]](https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96).
# * Best result is saved by using [ModelCheckpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint).

# In[ ]:


# https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96

import matplotlib.pyplot as plt

LR_START = 0.00001
LR_MAX = 0.00005 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = .8

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)

rng = [i for i in range(25 if EPOCHS<25 else EPOCHS)]
y = [lrfn(x) for x in rng]
plt.figure(figsize=(10, 4))
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))


# In[ ]:


from tensorflow.keras.callbacks import ModelCheckpoint

best_model_file_path = "best_model.hdf5"

def make_model_check_point(best_model_file_path):
    return ModelCheckpoint(
        best_model_file_path, monitor='val_auc', mode='max',
        verbose=0, save_best_only=True, save_weights_only=False, period=1)


# In[ ]:


model_check_point = make_model_check_point(best_model_file_path)

history = model.fit(
    get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS, validation_data=get_validation_dataset(),
    callbacks=[lr_callback, model_check_point],
    verbose=1)


# In[ ]:


model.load_weights(best_model_file_path)
eval_result = model.evaluate(get_validation_dataset(), verbose=0)
print(eval_result)


# <a name="Evaluation"></a>
# # Evaluation
# 
# <a name="LossAucPrecisionRecall"></a>
# ## Loss, Auc, Precision, Recall

# In[ ]:


def display_training_curve(training, validation, title, subplot, ylim):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    ax.set_ylim(*ylim)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])


# In[ ]:


display_training_curve(
    history.history['loss'], history.history['val_loss'],
    'loss', 221, (0.0, 0.2))
display_training_curve(
    history.history['auc'], history.history['val_auc'],
    'auc', 222, (0.5, 1.0))
display_training_curve(
    history.history['precision'], history.history['val_precision'],
    'precision', 223, (0.0, 1.0))
display_training_curve(
    history.history['recall'], history.history['val_recall'],
    'recall', 224, (0.0, 1.0))


# <a name="Roc"></a>
# ## Roc

# In[ ]:


def get_image(image, target):
    return image

val_image_ds = get_validation_dataset().map(get_image)


# In[ ]:


y_true = np.array([
    target.numpy() for _, target in iter(get_validation_dataset().unbatch()) ])
y_pred = model.predict(val_image_ds).flatten()

print(y_true.shape)
print(y_pred.shape)


# In[ ]:


from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_true, y_pred)

plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")
plt.show()


# In[ ]:


from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_true, y_pred)
print("AUC: {0:.3f}".format(auc))


# <a name="ConfusionMatrix"></a>
# ## Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('True Negatives: ', cm[0][0])
    print('False Positives: ', cm[0][1])
    print('False Negatives: ', cm[1][0])
    print('True Positives: ', cm[1][1])
    print('Total Positive Images: ', np.sum(cm[1]))


# In[ ]:


plot_cm(y_true, y_pred, p=0.5)


# <a name="Predictions"></a>
# # Predictions

# In[ ]:


# since we are splitting the dataset and iterating separately on images and ids, order matters.
test_ds = get_test_dataset(ordered=True)

print('Computing predictions...')
test_images_ds = test_ds.map(lambda image, image_name: image)
predictions = model.predict(test_images_ds).flatten()
print(predictions)

print('Generating submission.csv file...')
test_image_names_ds = test_ds.map(lambda image, image_name: image_name).unbatch()
test_image_names = np.array([
    image_name.numpy() for image_name in iter(test_image_names_ds) ]).astype('U')
np.savetxt(
    'submission.csv', np.rec.fromarrays([test_image_names, predictions]),
    fmt=['%s', '%f'], delimiter=',', header='image_name,target', comments='')
get_ipython().system('head submission.csv')


# Thanks for reading!
