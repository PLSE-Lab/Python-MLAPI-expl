#!/usr/bin/env python
# coding: utf-8

# ## About this kernel
# 
# Credited to @Nayu's kernel. 
# 
# I changed it with tf2.x codes and support TPU. For fast training, I resized the images into 64*64. <br>
# Let your TPU burn...

# I refered following kernels, thank you!
# 
# https://www.kaggle.com/ateplyuk/inat2019-starter-keras-efficientnet/data
# 
# https://www.kaggle.com/mobassir/keras-efficientnetb2-for-classifying-cloud
# 
# https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu
# 

# **Example of Fine-tuning from pretrained model using Keras  and Efficientnet (https://pypi.org/project/efficientnet/).**

# In[ ]:


import os, glob
import random
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import pandas as pd
import multiprocessing
from copy import deepcopy
from sklearn.metrics import precision_recall_curve, auc
from kaggle_datasets import KaggleDatasets
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, applications
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
from IPython.display import Image
from tqdm import tqdm_notebook as tqdm
import json
import os
import gc
from numpy.random import seed
seed(10)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system('pip install git+https://github.com/qubvel/efficientnet -q')


# # TPU setup

# In[ ]:


os.listdir('../input')


# In[ ]:



AUTO = tf.data.experimental.AUTOTUNE
try:
    # Create strategy from tpu
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print('tpu:',tpu)
except:
    print('no tpu.....')
    strategy=None
    tpu=None



# Data access
if tpu:
    GCS_DS_PATH = KaggleDatasets().get_gcs_path('iwildcam-2020-fgvc7')
    tf_records_path= KaggleDatasets().get_gcs_path('iwildcam2020-64-tf-records')
    tf_records_path2=KaggleDatasets().get_gcs_path('iwildcam2020-64-tf-records')
    test_tf_records_path=KaggleDatasets().get_gcs_path('iwildcam2020-64-tf-records')
    #If you want to use the best speed of TPU,then the batch_size should be times of 16. 
    #Since TPU V3-8 has 8 cores,so 16*8
    BATCH_SIZE = 16 * strategy.num_replicas_in_sync

# Configuration
EPOCHS = 8#3
img_size = 64#96


# In[ ]:


TRAINING_FILENAMES = tf.io.gfile.glob(tf_records_path + '/*train.rec')
TRAINING_FILENAMES.extend(tf.io.gfile.glob(tf_records_path2 + '/*zero.rec'))
TRAINING_FILENAMES


# In[ ]:


TEST_FILENAMES=tf.io.gfile.glob(test_tf_records_path + '/*test.rec')


# # Train data

# ## Data processing functions:

# In[ ]:



def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [img_size,img_size, 3]) # explicit size needed for TPU
    return image

def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
#     image = tf.image.random_flip_up_down(image)
    
    if label is None:
        return image
    else:
        return image, label
    
    
def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
    return image, label # returns a dataset of (image, label) pairs

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        # class is missing, this competitions's challenge is to predict flower classes for the test dataset
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['id']
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


# In[ ]:


train_dataset=get_training_dataset()


# In[ ]:


sub_df = pd.read_csv('../input/iwildcam-2020-fgvc7/sample_submission.csv')
sub_df.head()


# In[ ]:


test_dataset=get_test_dataset()


# ### Model

# In[ ]:


# import efficientnet.keras as efn 
import  efficientnet.tfkeras as efn 


with strategy.scope():
    model = tf.keras.Sequential([
        efn.EfficientNetB7(
            input_shape=(img_size, img_size, 3),
            weights='imagenet',
            include_top=False
        ),
        L.GlobalAveragePooling2D(),
        L.Dense(216, activation='softmax')#573
    ])

    model.compile(
        optimizer='adam',
         loss = 'sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    model.summary()


# In[ ]:


# Callbacks

early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
best_checkpoint='model.h5'
checkpoint = ModelCheckpoint(
    best_checkpoint, 
    monitor='val_accuracy', 
    verbose=1, 
#     save_best_only=True, 
    save_weights_only=True,
    mode='auto'
)


# # Train

# In[ ]:


get_ipython().run_cell_magic('time', '', '# STEPS_PER_EPOCH = train_x.shape[0] // BATCH_SIZE\nSTEPS_PER_EPOCH=(143736+3709) // BATCH_SIZE\nhistory = model.fit(\n    train_dataset, \n    epochs=EPOCHS, \n    verbose=2,\n    callbacks=[early,checkpoint],\n    steps_per_epoch=STEPS_PER_EPOCH,\n#     validation_data=valid_dataset\n)')


# In[ ]:


import gc

gc.collect()


# ### Test data

# In[ ]:


sam_sub_df=sub_df.copy()
sam_sub_df["file_name"] = sam_sub_df["Id"].map(lambda str : str + ".jpg")


# ### Prediction

# In[ ]:


get_ipython().run_cell_magic('time', '', 'model.load_weights(best_checkpoint)\n\npredict=model.predict(test_dataset, verbose=1).astype(float)')


# In[ ]:


print(len(predict))


# In[ ]:


predicted_class_indices=np.argmax(predict,axis=1)


# In[ ]:


predicted_class_indices


# In[ ]:


import pickle
with open('../input/iwildcam2020-classes-dict/cid_invert_dict.pkl', mode='rb') as fin:
    cid_invert_dict=pickle.load(fin)


# In[ ]:


def transform(x):
    return cid_invert_dict[str(x)]


# In[ ]:


sam_sub_df["Category"] = predicted_class_indices
sam_sub_df["Category"]=sam_sub_df["Category"].apply(transform)


         
sam_sub_df = sam_sub_df.loc[:,["Id", "Category"]]
sam_sub_df.to_csv("submission.csv",index=False)
sam_sub_df.head()


# In[ ]:





# In[ ]:




