#!/usr/bin/env python
# coding: utf-8

# ### imports

# # Introduction 
# This is my effort to do a *minimum* `Keras` replication with comparable baseline to the great kernel of @mhiro2 https://www.kaggle.com/mhiro2/simple-2d-cnn-classifier-with-pytorch (and further improved by @peining), which in turns use the excellent pre-processed data of @daisukelab https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data) -- Note that to inference to the private data in stage-2, you have to preprocess data yourself.
# 
# One change I made in a Keras version, instead of a simple conv net, I decide to use a pre-defined architectures [trained from scratch] `MobileNetV2`, `InceptionV3` and `Xception` where you can choose in the kernel. Also, many ideas borrow from a nice kernel of @voglinio https://www.kaggle.com/voglinio/keras-2d-model-5-fold-log-specgram-curated-only , I also borrow the SoftMax+BCE loss & TTA ideas from Giba's kernel (BTW, we all know Giba without having to mention his user :).
# 
# **UPDATE in V.17 : I add a simple CNN almost exactly the same as the pytorch baseline**
# 
# I apologize that my code is not at all clean; some of the `pytorch` code is still here albeit not used.
# 
# ## Major Updates
# * V1 [CV680, LB574]
# * V4 [CV66x, LB576]
# * V5 [] Add image augmentation module
# * V9 [CV679] Add lwlrap TF metric (credit @rio114 : https://www.kaggle.com/rio114/keras-cnn-with-lwlrap-evaluation )
# * V11 [] Employ list of augmentations mentioned in https://github.com/sainathadapa/kaggle-freesound-audio-tagging/blob/master/approaches_all.md
# * V16 [] Add BCEwithLogits (use only with ACTIVATION = 'linear')
# * V17 add SimpleCNN similar to the pytorch baseline
# * V20 add Curated-Only, Train-augment options
# 
# 
# **with BCEwithLogits and SimpleCNN, now this kernel should almost comparable to the pytorch baseline**
# 
# ### Minor Updates
# * V15[CV662]
# 

# In[3]:


import gc
import os
import pickle
import random
import time
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from psutil import cpu_count
import matplotlib.pyplot as plt

import librosa
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
#from skmultilearn.model_selection import iterative_train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastprogress import master_bar, progress_bar
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


# ### utils

# In[4]:


NUM_CLASSES = 80
checkpoint_file = 'model_best.h5'
SIZE=128
EPOCHS = 200 #150 for inception, 100 for xception
BATCH_SIZE = 64

LR = 4e-4
TTA = 19 #Number of test-time augmentation
PATIENCE = 5  #ReduceOnPlateau option
LR_FACTOR = 0.25 #ReduceOnPlateau option
CURATED_ONLY = True # use only curated data for training
TRAIN_AUGMENT = True # use augmentation for training data?
MODEL = 'inception' # choose among 'xception', 'inception', 'mobile', 'simple'

# if use BCEwithLogits loss, use Activation = 'linear' only
ACTIVATION = 'linear' 
# ACTIVATION = 'softmax'
# ACTIVATION = 'sigmoid'

# LOSS = 'categorical_crossentropy'
# LOSS = 'binary_crossentropy' 
LOSS = 'BCEwithLogits' 


# In[5]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 520
seed_everything(SEED)


# In[6]:


# from official code https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8#scrollTo=cRCaCIb9oguU
def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      truth: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.  Then the overall unbalanced lwlrap is
        simply np.sum(per_class_lwlrap * weight_per_class)
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    # Space to store a distinct precision value for each class on each sample.
    # Only the classes that are true for each sample will be filled in.
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = (
            _one_sample_positive_class_precisions(scores[sample_num, :],
                                                  truth[sample_num, :]))
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
            precision_at_hits)
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    # Form average of each column, i.e. all the precisions assigned to labels in
    # a particular class.
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
    #                = np.sum(per_class_lwlrap * weight_per_class)
    return per_class_lwlrap, weight_per_class


# In[7]:


import tensorflow as tf

# from https://www.kaggle.com/ratthachat/keras-cnn-with-lwlrap-evaluation/edit
def tf_one_sample_positive_class_precisions(y_true, y_pred) :
    num_samples, num_classes = y_pred.shape
    
    # find true labels
    pos_class_indices = tf.where(y_true > 0) 
    
    # put rank on each element
    retrieved_classes = tf.nn.top_k(y_pred, k=num_classes).indices
    sample_range = tf.zeros(shape=tf.shape(tf.transpose(y_pred)), dtype=tf.int32)
    sample_range = tf.add(sample_range, tf.range(tf.shape(y_pred)[0], delta=1))
    sample_range = tf.transpose(sample_range)
    sample_range = tf.reshape(sample_range, (-1,num_classes*tf.shape(y_pred)[0]))
    retrieved_classes = tf.reshape(retrieved_classes, (-1,num_classes*tf.shape(y_pred)[0]))
    retrieved_class_map = tf.concat((sample_range, retrieved_classes), axis=0)
    retrieved_class_map = tf.transpose(retrieved_class_map)
    retrieved_class_map = tf.reshape(retrieved_class_map, (tf.shape(y_pred)[0], num_classes, 2))
    
    class_range = tf.zeros(shape=tf.shape(y_pred), dtype=tf.int32)
    class_range = tf.add(class_range, tf.range(num_classes, delta=1))
    
    class_rankings = tf.scatter_nd(retrieved_class_map,
                                          class_range,
                                          tf.shape(y_pred))
    
    #pick_up ranks
    num_correct_until_correct = tf.gather_nd(class_rankings, pos_class_indices)

    # add one for division for "presicion_at_hits"
    num_correct_until_correct_one = tf.add(num_correct_until_correct, 1) 
    num_correct_until_correct_one = tf.cast(num_correct_until_correct_one, tf.float32)
    
    # generate tensor [num_sample, predict_rank], 
    # top-N predicted elements have flag, N is the number of positive for each sample.
    sample_label = pos_class_indices[:, 0]   
    sample_label = tf.reshape(sample_label, (-1, 1))
    sample_label = tf.cast(sample_label, tf.int32)
    
    num_correct_until_correct = tf.reshape(num_correct_until_correct, (-1, 1))
    retrieved_class_true_position = tf.concat((sample_label, 
                                               num_correct_until_correct), axis=1)
    retrieved_pos = tf.ones(shape=tf.shape(retrieved_class_true_position)[0], dtype=tf.int32)
    retrieved_class_true = tf.scatter_nd(retrieved_class_true_position, 
                                         retrieved_pos, 
                                         tf.shape(y_pred))
    # cumulate predict_rank
    retrieved_cumulative_hits = tf.cumsum(retrieved_class_true, axis=1)

    # find positive position
    pos_ret_indices = tf.where(retrieved_class_true > 0)

    # find cumulative hits
    correct_rank = tf.gather_nd(retrieved_cumulative_hits, pos_ret_indices)  
    correct_rank = tf.cast(correct_rank, tf.float32)

    # compute presicion
    precision_at_hits = tf.truediv(correct_rank, num_correct_until_correct_one)

    return pos_class_indices, precision_at_hits

def tf_lwlrap(y_true, y_pred):
    num_samples, num_classes = y_pred.shape
    pos_class_indices, precision_at_hits = (tf_one_sample_positive_class_precisions(y_true, y_pred))
    pos_flgs = tf.cast(y_true > 0, tf.int32)
    labels_per_class = tf.reduce_sum(pos_flgs, axis=0)
    weight_per_class = tf.truediv(tf.cast(labels_per_class, tf.float32),
                                  tf.cast(tf.reduce_sum(labels_per_class), tf.float32))
    sum_precisions_by_classes = tf.zeros(shape=(num_classes), dtype=tf.float32)  
    class_label = pos_class_indices[:,1]
    sum_precisions_by_classes = tf.unsorted_segment_sum(precision_at_hits,
                                                        class_label,
                                                       num_classes)
    labels_per_class = tf.cast(labels_per_class, tf.float32)
    labels_per_class = tf.add(labels_per_class, 1e-7)
    per_class_lwlrap = tf.truediv(sum_precisions_by_classes,
                                  tf.cast(labels_per_class, tf.float32))
    out = tf.cast(tf.tensordot(per_class_lwlrap, weight_per_class, axes=1), dtype=tf.float32)
    return out


# In[8]:


from keras import backend as k
def BCEwithLogits(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred, from_logits=True), axis=-1)


# ### dataset

# In[9]:


dataset_dir = Path('../input/freesound-audio-tagging-2019')
preprocessed_dir = Path('../input/fat2019_prep_mels1')


# In[10]:


csvs = {
    'train_curated': dataset_dir / 'train_curated.csv',
    #'train_noisy': dataset_dir / 'train_noisy.csv',
    'train_noisy': preprocessed_dir / 'trn_noisy_best50s.csv',
    'sample_submission': dataset_dir / 'sample_submission.csv',
}

dataset = {
    'train_curated': dataset_dir / 'train_curated',
    'train_noisy': dataset_dir / 'train_noisy',
    'test': dataset_dir / 'test',
}

mels = {
    'train_curated': preprocessed_dir / 'mels_train_curated.pkl',
    'train_noisy': preprocessed_dir / 'mels_trn_noisy_best50s.pkl',
    'test': preprocessed_dir / 'mels_test.pkl',  # NOTE: this data doesn't work at 2nd stage
}


# In[11]:


train_curated = pd.read_csv(csvs['train_curated'])
train_noisy = pd.read_csv(csvs['train_noisy'])
if CURATED_ONLY:
    train_df = train_curated
else:
    train_df = pd.concat([train_curated, train_noisy], sort=True, ignore_index=True)
train_df.head()


# In[12]:


test_df = pd.read_csv(csvs['sample_submission'])
test_df.head()


# In[13]:


labels = test_df.columns[1:].tolist()
labels[:10]


# In[14]:


num_classes = len(labels)
num_classes


# In[15]:


y_train = np.zeros((len(train_df), num_classes)).astype(int)
for i, row in enumerate(train_df['labels'].str.split(',')):
    for label in row:
        idx = labels.index(label)
        y_train[i, idx] = 1

y_train.shape


# In[16]:


with open(mels['train_curated'], 'rb') as curated, open(mels['train_noisy'], 'rb') as noisy:
    x_train = pickle.load(curated)
    if CURATED_ONLY == False:
        x_train.extend(pickle.load(noisy))

with open(mels['test'], 'rb') as test:
    x_test = pickle.load(test)
    
len(x_train), len(x_test)


# In[17]:



for ii in range(5):
    print(x_train[ii].shape) #x_train is of shape (TRAIN_NUM,128,LEN,3) [4D Tensor]
    print(x_test[ii].shape,'\n')  #x_test of shape (TEST_NUM,128,LEN,3) [4D Tensor]


# ### model

# In[18]:


from keras.layers import *
from keras.models import Sequential, load_model, Model
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
import keras
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_inception
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobile
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as preprocess_xception

from keras.utils import Sequence
from sklearn.utils import shuffle
def create_model_inception(n_out=NUM_CLASSES):

    base_model =InceptionV3(weights=None, include_top=False)
    
    x0 = base_model.output
    x1 = GlobalAveragePooling2D()(x0)
    x2 = GlobalMaxPooling2D()(x0)
    x = Concatenate()([x1,x2])
    
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    
    predictions = Dense(n_out, activation=ACTIVATION)(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


# In[19]:


def create_model_xception(n_out=NUM_CLASSES):

    base_model = Xception(weights=None, include_top=False)
    
    x0 = base_model.output
    x1 = GlobalAveragePooling2D()(x0)
    x2 = GlobalMaxPooling2D()(x0)
    x = Concatenate()([x1,x2])
    
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

#     x = Dense(128, activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
    
    predictions = Dense(n_out, activation=ACTIVATION)(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


# In[20]:


def create_model_mobile(n_out=NUM_CLASSES):

    base_model =MobileNetV2(weights=None, include_top=False)
    
    x0 = base_model.output
    x1 = GlobalAveragePooling2D()(x0)
    x2 = GlobalMaxPooling2D()(x0)
    x = Concatenate()([x1,x2])
    
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

#     x = Dense(128, activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.25)(x)

    
    predictions = Dense(n_out, activation=ACTIVATION)(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


# In[21]:


def conv_simple_block(x, n_filters):
    
    x = Convolution2D(n_filters, (3,1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Convolution2D(n_filters, (3,1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = AveragePooling2D()(x)

    return x

def create_model_simplecnn(n_out=NUM_CLASSES):
    
    inp = Input(shape=(128,128,3))
#     inp = Input(shape=(None,None,3))
    x = conv_simple_block(inp,64)
    x = conv_simple_block(x,128)
    x = conv_simple_block(x,256)
    x = conv_simple_block(x,512)
    
    x1 = GlobalAveragePooling2D()(x)
    x2 = GlobalMaxPooling2D()(x)
    x = Add()([x1,x2])

    x = Dropout(0.2)(x)

    x = Dense(128, activation='linear')(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    predictions = Dense(n_out, activation=ACTIVATION)(x)

    model = Model(inputs=inp, outputs=predictions)
    return model


# In[22]:


'''Choose your model here'''
if MODEL == 'xception':
    preprocess_input = preprocess_xception
    model = create_model_xception(n_out=NUM_CLASSES)
if MODEL == 'inception':
    preprocess_input = preprocess_inception
    model = create_model_inception(n_out=NUM_CLASSES)
if MODEL == 'mobile':
    preprocess_input = preprocess_mobile
    model = create_model_mobile(n_out=NUM_CLASSES)
else:
    preprocess_input = preprocess_mobile
    model = create_model_simplecnn(
    n_out=NUM_CLASSES)

print(MODEL)
model.summary()


# ### train

# In[23]:


# If you want, you can try more advanced augmentation like this
augment_img = iaa.Sequential([
    iaa.SomeOf((0,3),[
#         iaa.ContrastNormalization((0.9, 1.1)),
#         iaa.Multiply((0.9, 1.1), per_channel=0.2),
        iaa.Fliplr(0.5),
        iaa.GaussianBlur(sigma=(0, 0.1)),
        iaa.Affine( # x-shift
            translate_percent={"x": (-0.1, 0.1), "y": (-0.0, 0.0)},
        ),
        iaa.CoarseDropout(0.1,size_percent=0.05) # see examples : https://github.com/aleju/imgaug
            ])], random_order=True)


# Or you can choose this simplest augmentation (like pytorch version)
# augment_img = iaa.Fliplr(0.5)

# This is my ugly modification; sorry about that
class FATTrainDataset(Sequence):

    def getitem(image):
        # crop 2sec

        base_dim, time_dim, _ = image.shape
        crop = random.randint(0, time_dim - base_dim)
        image = image[:,crop:crop+base_dim,:]

        image = preprocess_input(image)
        
#         label = self.labels[idx]
        return image
    def create_generator(train_X, train_y, batch_size, shape, augument=False, shuffling=False, test_data=False):
        assert shape[2] == 3
        while True:
            if shuffling:
                train_X,train_y = shuffle(train_X,train_y)

            for start in range(0, len(train_y), batch_size):
                end = min(start + batch_size, len(train_y))
                batch_images = []
                X_train_batch = train_X[start:end]
                if test_data == False:
                    batch_labels = train_y[start:end]
                
                for i in range(len(X_train_batch)):
                    image = FATTrainDataset.getitem(X_train_batch[i])   
                    if augument:
                        image = FATTrainDataset.augment(image)
                    batch_images.append(image)
                    
                if test_data == False:
                    yield np.array(batch_images, np.float32), batch_labels
                else:
                    yield np.array(batch_images, np.float32)
        return image
    
    def augment(image):

        image_aug = augment_img.augment_image(image)
        return image_aug


# In[24]:


from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,
                             EarlyStopping, ReduceLROnPlateau,CSVLogger)
                             
from sklearn.model_selection import train_test_split

checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_tf_lwlrap', verbose=1, 
                             save_best_only=True, mode='max', save_weights_only = False)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_tf_lwlrap', factor=LR_FACTOR, patience=PATIENCE, 
                                   verbose=1, mode='max', min_delta=0.0001, cooldown=2, min_lr=1e-5 )

csv_logger = CSVLogger(filename='../working/training_log.csv',
                       separator=',',
                       append=True)


# split data into train, valid
x_trn, x_val, y_trn, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=SEED)

# create train and valid datagens
train_generator = FATTrainDataset.create_generator(
    x_trn, y_trn, BATCH_SIZE, (SIZE,SIZE,3), augument=TRAIN_AUGMENT, shuffling=True)
validation_generator = FATTrainDataset.create_generator(
    x_val, y_val, BATCH_SIZE, (SIZE,SIZE,3), augument=False, shuffling=False)

callbacks_list = [checkpoint, csv_logger, reduceLROnPlat]


# In[25]:


train_steps = np.ceil(float(len(x_trn)) / float(BATCH_SIZE))
val_steps = np.ceil(float(len(x_val)) / float(BATCH_SIZE))
train_steps = train_steps.astype(int)
val_steps = val_steps.astype(int)
print(train_steps, val_steps)
print(len(x_trn), BATCH_SIZE)


# In[26]:


print(LOSS)
if LOSS=='BCEwithLogits':
     model.compile(loss=BCEwithLogits,
            optimizer=Adam(lr=LR),
            metrics=[tf_lwlrap,'categorical_accuracy'])
else:
    model.compile(loss=LOSS,
            optimizer=Adam(lr=LR),
            metrics=[tf_lwlrap,'categorical_accuracy'])


# In[27]:




hist = model.fit_generator(
    train_generator,
    steps_per_epoch=train_steps,
    validation_data=validation_generator,
    validation_steps=val_steps,
    epochs=EPOCHS,
    verbose=1,
    callbacks=callbacks_list)


# In[28]:


print(K.eval(model.optimizer.lr))


# In[29]:


fig, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].set_title('loss')
ax[0].plot(hist.epoch, hist.history["loss"], label="Train loss")
ax[0].plot(hist.epoch, hist.history["val_loss"], label="Validation loss")
ax[1].set_title('categorical_accuracy')
ax[1].plot(hist.epoch, hist.history["categorical_accuracy"], label="Train categorical_accuracy")
ax[1].plot(hist.epoch, hist.history["val_categorical_accuracy"], label="Validation categorical_accuracy")
ax[0].legend()
ax[1].legend()


# In[30]:


fig, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].set_title('tf_lwlrap')
ax[0].plot(hist.epoch, hist.history["tf_lwlrap"], label="Train lwlrap")
ax[0].plot(hist.epoch, hist.history["val_tf_lwlrap"], label="Validation lwlrap")
ax[1].set_title('categorical_accuracy')
ax[1].plot(hist.epoch, hist.history["categorical_accuracy"], label="Train categorical_accuracy")
ax[1].plot(hist.epoch, hist.history["val_categorical_accuracy"], label="Validation categorical_accuracy")
ax[0].legend()
ax[1].legend()


# In[31]:


model.load_weights(checkpoint_file)


# # Calculate Validation Score using TTA
# Note that we have to initiate validation_generation everytime before doing a new prediction as `model.fit_generator` will mis-index examples at the end of epoch (and you will get random score)

# In[32]:


validation_generator = FATTrainDataset.create_generator(
    x_val, y_val, BATCH_SIZE, (SIZE,SIZE,3), augument=False, shuffling=False)

pred_val_y = model.predict_generator(validation_generator,steps=val_steps,verbose=1)
for ii in range(TTA):
    validation_generator = FATTrainDataset.create_generator(
        x_val, y_val, BATCH_SIZE, (SIZE,SIZE,3), augument=False, shuffling=False)

    pred_val_y += model.predict_generator(validation_generator,steps=val_steps,verbose=1)

'''Since the score is based on ranking, we do not need to normalize the prediction'''
# pred_val_y = pred_val_y/10


# In[33]:


train_generator = FATTrainDataset.create_generator(
    x_trn, y_trn, BATCH_SIZE, (SIZE,SIZE,3), augument=True, shuffling=False)
pred_train_y = model.predict_generator(train_generator,steps=train_steps,verbose=1)


# In[34]:


import sklearn.metrics
def calculate_overall_lwlrap_sklearn(truth, scores):
    """Calculate the overall lwlrap using sklearn.metrics.lrap."""
    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
    sample_weight = np.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = sklearn.metrics.label_ranking_average_precision_score(
      truth[nonzero_weight_sample_indices, :] > 0, 
      scores[nonzero_weight_sample_indices, :], 
      sample_weight=sample_weight[nonzero_weight_sample_indices])
    return overall_lwlrap


# In[35]:


print(pred_val_y.shape, y_val.shape)
print(np.sum(pred_val_y), np.sum(y_val))
# for ii in range(len(y_val)):
#     print(np.sum(pred_val_y[ii]), np.sum(y_val[ii]))


# In[36]:


print("lwlrap from sklearn.metrics for training data =", calculate_overall_lwlrap_sklearn(y_trn, pred_train_y))
print("lwlrap from sklearn.metrics =", calculate_overall_lwlrap_sklearn(y_val, pred_val_y/10))

score, weight = calculate_per_class_lwlrap(y_val, pred_val_y)
lwlrap = (score * weight).sum()
print('direct calculation of lwlrap : %.4f' % (lwlrap))


# ## Predict Test Data with TTA

# In[ ]:





# In[37]:


test_steps = np.ceil(float(len(x_test)) / float(BATCH_SIZE)).astype(int)


# In[38]:


test_generator = FATTrainDataset.create_generator(
    x_test, x_test, BATCH_SIZE, (SIZE,SIZE,3), augument=False, shuffling=False, test_data=True)
pred_test_y = model.predict_generator(test_generator,steps=test_steps,verbose=1)

for ii in range(TTA):
    test_generator = FATTrainDataset.create_generator(
        x_test, x_test, BATCH_SIZE, (SIZE,SIZE,3), augument=False, shuffling=False, test_data=True)
    pred_test_y += model.predict_generator(test_generator,steps=test_steps,verbose=1)


# In[39]:


sort_idx = np.argsort(labels).astype(int)


# In[40]:


print(sort_idx)


# In[41]:


sample_sub = pd.read_csv('../input/freesound-audio-tagging-2019/sample_submission.csv')
test_Y_sort = pred_test_y[:, sort_idx]
sample_sub.iloc[:, 1:] =  test_Y_sort
sample_sub.to_csv('submission.csv', index=False)

sample_sub.head()
