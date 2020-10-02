#!/usr/bin/env python
# coding: utf-8

# # I WILL CREDIT https://www.kaggle.com/ajax0564(ANKIT MAURYA) FOR THE TRAINING AND MODEL

# In[ ]:


get_ipython().system('pip install -q efficientnet')

import os
import re

import numpy as np
import pandas as pd
import math

from matplotlib import pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras.layers as L
import efficientnet.tfkeras as efn



from kaggle_datasets import KaggleDatasets


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# v1
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

# v3
import warnings
warnings.filterwarnings('ignore')

from sklearn.utils import shuffle

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import LearningRateScheduler
from keras.metrics import *
# v4

ACCURACY_LIST = []
from keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalMaxPooling2D
from keras.models import Model



# v6
# Get reproducible results
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(1)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:





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


# In[ ]:


data_dir = '/kaggle/input/siim-isic-melanoma-classification/jpeg/'
train_path = data_dir + '/train/'
test_path = data_dir + '/test/'


# In[ ]:


from matplotlib.image import imread


# In[ ]:


train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')


# In[ ]:


train.sample(10)


# In[ ]:


def append_ext(fn):
    return train_path + fn+ '.jpg'
train["image_name"]=train["image_name"].apply(append_ext)


# In[ ]:


print(f"Count of null values in train :\n{train.isnull().sum()}")


# In[ ]:


sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# In[ ]:


sns.countplot(x='sex', data=train)


# In[ ]:


sns.countplot(x='benign_malignant', data=train)


# # THIS SHOWS THE IMBALANCE IN THE DATASET

# In[ ]:


sns.countplot(x='target', data=train)


# In[ ]:


train['age_approx']= train['age_approx'].fillna(0)


# In[ ]:


mean = train['age_approx'].mean()


# In[ ]:


def impute_age(cols):
    x = cols[0]
    if(x==0):
        return mean
    else:
        return x


# In[ ]:


train['age_approx'] = train[['age_approx']].apply(impute_age, axis=1)


# In[ ]:


train['anatom_site_general_challenge'].unique()


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x='anatom_site_general_challenge', data=train)


# In[ ]:


train['anatom_site_general_challenge']= train['anatom_site_general_challenge'].fillna('torso')


# In[ ]:


train['sex']= train['sex'].fillna('male')


# In[ ]:


a = os.listdir(train_path)[0]


# In[ ]:


sam_image = train_path+a


# In[ ]:


sam_img_tensor = imread(sam_image)


# In[ ]:


sam_img_tensor.shape


# In[ ]:


plt.imshow(sam_img_tensor)


# In[ ]:


len(os.listdir(train_path))


# # **Image Histograms**
# 
# 
# An image histogram is a type of histogram that acts as a graphical representation of the tonal distribution in a digital image. It plots the number of pixels for each tonal value. By looking at the histogram for a specific image a viewer will be able to judge the entire tonal distribution at a glance.

# In[ ]:


fig, ax = plt.subplots(4, 2, figsize=(20, 20))

malignant_file_paths = train[train['benign_malignant'] == 'malignant']['image_name'].values
sample_file_paths = malignant_file_paths[:4]
sample_covid19_file_paths = list(map(lambda x: os.path.join(train_path, x), sample_file_paths))

for row, file_path in enumerate(sample_file_paths):
    image = plt.imread(file_path)
    ax[row, 0].imshow(image)
    ax[row, 1].hist(image.ravel(), 256, [0,256])
    ax[row, 0].axis('off')
    if row == 0:
        ax[row, 0].set_title('Images')
        ax[row, 1].set_title('Histograms')
fig.suptitle('Label Malignant', size=16)
plt.show()


# In[ ]:


fig, ax = plt.subplots(4, 2, figsize=(20, 20))

malignant_file_paths = train[train['benign_malignant'] == 'benign']['image_name'].values
sample_file_paths = malignant_file_paths[:4]
sample_covid19_file_paths = list(map(lambda x: os.path.join(train_path, x), sample_file_paths))

for row, file_path in enumerate(sample_file_paths):
    image = plt.imread(file_path)
    ax[row, 0].imshow(image)
    ax[row, 1].hist(image.ravel(), 256, [0,256])
    ax[row, 0].axis('off')
    if row == 0:
        ax[row, 0].set_title('Images')
        ax[row, 1].set_title('Histograms')
fig.suptitle('Label Benign', size=16)
plt.show()


# In[ ]:


train['diagnosis'].unique()


# In[ ]:


plt.figure(figsize=(15,6))
sns.countplot(x='diagnosis', data=train)


# In[ ]:


train.drop('sex', axis=1, inplace=True)


# In[ ]:


train.drop('patient_id', axis=1, inplace=True)


# In[ ]:


train.drop('anatom_site_general_challenge', axis=1, inplace=True)


# In[ ]:


train.drop('diagnosis', axis=1, inplace=True)


# In[ ]:


train.drop('age_approx', axis=1, inplace=True)


# In[ ]:


train.drop('target', axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:





# In[ ]:


AUTO = tf.data.experimental.AUTOTUNE

# Data access
GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')

# Configuration
EPOCHS = 7
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
IMAGE_SIZE = [1024, 1024]
imSize = 512


# In[ ]:


def append_path(pre):
    return np.vectorize(lambda file: os.path.join(GCS_DS_PATH, pre, file))


# In[ ]:


sub = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/train*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/test*.tfrec')


# In[ ]:


train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
train.head(1)


# In[ ]:


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3])# explicit size needed for TPU
    image = tf.image.resize(image, [imSize,imSize])
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

def data_augment(image, label):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_saturation(image, 0, 2)
    return image, label   

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

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print('Dataset: {} training images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_TEST_IMAGES))


# In[ ]:


from tensorflow.keras.applications import ResNet152V2


# In[ ]:


METRICS = [
      BinaryAccuracy(name='accuracy'),
      AUC(name='auc'),
]


# In[ ]:


with strategy.scope():
    effn = efn.EfficientNetB7(include_top=False,
        input_shape=(512,512, 3),
        weights='imagenet'
        
    )
    effn.trainable = True  
    
    
    model = tf.keras.Sequential([
        effn,
        L.GlobalAveragePooling2D(),
        L.Dense(1024),
        L.ELU(alpha=0.2),
        L.Dropout(0.4),
        
        
        L.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss = 'binary_crossentropy',
        metrics=[METRICS]
    )


# In[ ]:


def build_lrfn(lr_start=0.00001, lr_max=0.0001, 
               lr_min=0.000001, lr_rampup_epochs=20, 
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


# In[ ]:


lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)


# In[ ]:


test_ds = get_test_dataset(ordered=True)
# valid_dataset = get_validation_dataset(ordered=False)
train_dataset = get_training_dataset() 


# In[ ]:


STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
model.fit(
    train_dataset, 
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=15,
    callbacks=[lr_schedule])


# In[ ]:


model.save_weights('final_model_weights.h5')
model.save('final_model.h5')


# In[ ]:


lossess = pd.DataFrame(model.history.history)


# In[ ]:





# In[ ]:


lossess.head(7)


# In[ ]:


test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds,verbose = 1)


# In[ ]:


print('Generating submission.csv file...')
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')


# In[ ]:


pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities)})
pred_df.head(5)


# In[ ]:


del sub['target']
sub = sub.merge(pred_df, on='image_name')
sub.to_csv('submission.csv', index=False)
sub.head(5)


# In[ ]:




