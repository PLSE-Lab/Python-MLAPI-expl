#!/usr/bin/env python
# coding: utf-8

# Hello Everyone, 
# This notebook uses the data put together by [Alex Shonenkov](https://www.kaggle.com/shonenkov) and later on converted to **TFRecrods** format by [Chris Deotte](https://www.kaggle.com/cdeotte).  
# 
# ---
#  [Alex Shonenkov](https://www.kaggle.com/shonenkov) merged some of the well known melanoma detection datasets together [here](https://www.kaggle.com/shonenkov/melanoma-merged-external-data-512x512-jpeg). Then  [Chris Deotte](https://www.kaggle.com/cdeotte) converted the whole dataset into **TFRecords** format [here](https://www.kaggle.com/cdeotte/512x512-melanoma-tfrecords-70k-images). Please upvote them for their excellent dedication. 
#  
#  ### The four merged datasets are: 
#  
# - [Melanoma Detection Dataset](https://www.kaggle.com/wanderdust/skin-lesion-analysis-toward-melanoma-detection)
# - [Skin Lesion Images for Melanoma Classification](https://www.kaggle.com/andrewmvd/isic-2019)
# - [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)
# - [SIIM-ISIC Melanoma Classification](https://www.kaggle.com/c/siim-isic-melanoma-classification/data)
# ---
# 
# # Here, I just used basic EfficentNetB7 model to train and predict without any ensemble. 
# 
# # I will add several optimizations gradually.
# 
# 
# # Please upvote, if you like this kernel. 

# # 1. Loading Libraries

# In[ ]:


get_ipython().system('pip install -q efficientnet')
get_ipython().system('pip install -q pyyaml h5py')

#basic libraries
import os, re, math
import numpy as np
import pandas as pd

#plot libraries
import matplotlib.pyplot as plt
import plotly.express as px

#utilities library
from sklearn import metrics
from sklearn.model_selection import train_test_split

#background library for learning 
import tensorflow as tf
import tensorflow.keras.layers as Layers

from kaggle_datasets import KaggleDatasets

import efficientnet.tfkeras as efn


# In[ ]:


train_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
test_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
sub = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')


# # 4. TPU Setup Code

# In[ ]:


try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# # 5. Setting up Running Configuration 

# In[ ]:


# For tf.dataset Tensorflow tf. data AUTOTUNE. ... prefetch transformation, 
# which can be used to decouple the time when data is produced from the time when data is consumed. 
# In particular, the transformation uses a background thread and an internal buffer to prefetch 
# elements from the input dataset ahead of the time they are requested.
AUTO = tf.data.experimental.AUTOTUNE

# Get data access to the dataset for TPUs
GCS_PATH_ORIGINAL = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')
GCS_PATH_MERGED = KaggleDatasets().get_gcs_path('512x512-melanoma-tfrecords-70k-images')

# Running Configuration 
EPOCHS = 20
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
IMAGE_SIZE = [512, 512]

TRAINING = True

# Listing the filenames in TFRecords fomat
# TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH1 + '/tfrecords/train*.tfrec')
# TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH1 + '/tfrecords/test*.tfrec')

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH_MERGED + '/train*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH_MERGED + '/test*.tfrec')

CLASSES = [0,1]
print('Training filenames\n', list(TRAINING_FILENAMES))
print('Test file names\n', list(TEST_FILENAMES))


# # 6. Training Validation Split

# In[ ]:


# import random 
# random.shuffle(TRAINING_FILENAMES)
VALIDATION_FILENAMES = TRAINING_FILENAMES[0:5]
TRAINING_FILENAMES = TRAINING_FILENAMES[5:]


# In[ ]:


VALIDATION_FILENAMES


# # 6. Helper Functions

# In[ ]:


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        #"class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
        "target": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    #label = tf.cast(example['class'], tf.int32)
    label = tf.cast(example['target'], tf.int32)
    return image, label # returns a dataset of (image, label) pairs

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        # class is missing, this competitions's challenge is to predict flower classes for the test dataset
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['image_name']
    return image, idnum # returns a dataset of image(s)

def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset


def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)



def visualize_training(history, lw = 3):
    plt.figure(figsize=(10,6))
    plt.plot(history.history['accuracy'], label = 'training', marker = '*', linewidth = lw)
    plt.plot(history.history['val_accuracy'], label = 'validation', marker = 'o', linewidth = lw)
    plt.title('Training Accuracy vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(fontsize = 'x-large')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label = 'training', marker = '*', linewidth = lw)
    plt.plot(history.history['val_loss'], label = 'validation', marker = 'o', linewidth = lw)
    plt.title('Training Loss vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize = 'x-large')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(history.history['lr'], label = 'lr', marker = '*',linewidth = lw)
    plt.title('Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.show()


# # 7. Data Augmentation 

# In[ ]:


def data_augment(image, label):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (above),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    #image = tf.image.random_saturation(image, 0, 2)
    return image, label   


# # 8. Data Sumamry

# In[ ]:


NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
NUM_VALID_IMAGES = count_data_items(VALIDATION_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset Details:\nTraining images: {},  \nValidation Images: {} \nTest Images (unlabeled): {}'.format(NUM_TRAINING_IMAGES, NUM_VALID_IMAGES, NUM_TEST_IMAGES))


df = pd.DataFrame({'data':['NUM_TRAINING_IMAGES', 'NUM_TEST_IMAGES'],
                   'No of Samples':[NUM_TRAINING_IMAGES, NUM_TEST_IMAGES]})
plt.figure()
x = df.plot.bar(x='data', y='No of Samples', rot=0)
plt.ylabel('No of Samples')
plt.title('No of Training and Test Images')
plt.show()


# # 9. Learning Rate Scheduler

# In[ ]:


def build_lrfn(lr_start=0.00001, lr_max=0.000075, 
               lr_min=0.000001, lr_rampup_epochs=8, 
               lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr
    
    return lrfn

lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)


# In[ ]:


early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                                 patience=4,  
                                                 mode='max',
                                                 baseline=None, 
                                                 restore_best_weights=True)

model_checkpoint_callback_efnB7 = tf.keras.callbacks.ModelCheckpoint(
    filepath='model_efnB7_all_four_data_best_val_acc.hdf5',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


# # Getting Ready with the Test Dataset

# In[ ]:


test_ds = get_test_dataset(ordered=True)
print('Generating test file list')
test_images_ds = test_ds.map(lambda image, idnum: image)


# # 10. Training Models

# In[ ]:


with strategy.scope():
    model_efn_b7 = tf.keras.Sequential([
        efn.EfficientNetB7(
            input_shape=(*IMAGE_SIZE, 3),
            weights='imagenet',
            include_top=False
        ),
        Layers.GlobalAveragePooling2D(),
        Layers.Dense(1, activation='sigmoid')
    ])
    model_efn_b7.compile(
        optimizer='adam',
        loss = 'binary_crossentropy',
        metrics=['accuracy']
    )
    model_efn_b7.summary()

    
if TRAINING:
    history_efn_b7 = model_efn_b7.fit(
        get_training_dataset(), 
        epochs=EPOCHS, 
        callbacks=[lr_schedule, model_checkpoint_callback_efnB7, early_stopper],
        steps_per_epoch=NUM_TRAINING_IMAGES // BATCH_SIZE,
        validation_data=get_validation_dataset()
    )
    
if TRAINING:
    pd.DataFrame.from_dict(history_efn_b7.history).to_csv('history_efn_b7.csv' , index=False)
    visualize_training(history_efn_b7)
else:
    model_efn_b7.load_weights('/kaggle/input/efficent-net-pretrained-24-epoch/model_efnB7_all_four_data_best_val_acc.hdf5')
        
probabilities_efn_b7 = model_efn_b7.predict(test_images_ds)
# tf.tpu.experimental.initialize_tpu_system(tpu)
# visualize_training(history_efn_b7)


# # 11. Prediction and Submission Generation

# In[ ]:


test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') 
# all in one batch 


# In[ ]:


pred_efn_b7 = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities_efn_b7)})
pred_efn_b7.to_csv('submission_efn_b7_all_four_datasets.csv', index = False)


# In[ ]:


# pred_efn_b0 = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities_efn_b0)}) pred_efn_b7 = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities_efn_b7)}) pred_dnet201 = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities_dnet201)})

# mean_output = pred_efn_b0.copy() mean_output.target = pred_efn_b0.target *0.2 + pred_efn_b7.target * 0.7 + pred_dnet201.target * 0.1 mean_output.to_csv('mean_submission.csv', index = False)


# sub_pred_efn_b0 = sub.copy()
# sub_pred_efn_b7 = sub.copy()
# sub_pred_dnet201 = sub.copy()



# del sub_pred_efn_b0['target']
# sub_pred_efn_b0 = sub_pred_efn_b0.merge(pred_efn_b0, on='image_name')


# del sub_pred_efn_b7['target']
# sub_pred_efn_b7 = sub_pred_efn_b7.merge(pred_efn_b7, on='image_name')


# del sub_pred_dnet201['target']
# sub_pred_dnet201 = sub_pred_dnet201.merge(pred_dnet201, on='image_name')

# sub_pred_efn_b0.to_csv('submission_efn_b0.csv', index = False)
# sub_pred_efn_b7.to_csv('submission_efn_b7.csv', index = False)
# sub_pred_dnet201.to_csv('submission_dnet201.csv', index = False)
# sub_pred_efn_b0.target =(sub_pred_efn_b0.target + sub_pred_efn_b7.target +  sub_pred_dnet201.target)/3.0
# sub_pred_efn_b0.to_csv('submission.csv', index=False)


# # References: 
# * https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/155251
# * https://www.kaggle.com/redwankarimsony/power-of-metadata-xgboost-cnn-ensemble/
# * https://www.kaggle.com/cdeotte/512x512-melanoma-tfrecords-70k-images
# * https://www.kaggle.com/shonenkov/melanoma-merged-external-data-512x512-jpeg
# 
# 
# 
