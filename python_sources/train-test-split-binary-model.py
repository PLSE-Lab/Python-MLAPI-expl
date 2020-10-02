#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
from skimage.transform import resize
from imgaug import augmenters as iaa
import imgaug as ia
from tqdm import tqdm
import PIL
from PIL import Image, ImageOps
import cv2
from sklearn.utils import class_weight, shuffle
from keras.losses import binary_crossentropy
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import f1_score, fbeta_score, cohen_kappa_score, confusion_matrix, classification_report
from keras.utils import Sequence
from keras.utils import to_categorical
from keras import layers
from keras import models
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras import regularizers, optimizers
from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,
                             EarlyStopping, ReduceLROnPlateau,CSVLogger)
from keras.layers import Dropout
from keras.utils import np_utils
from keras.losses import binary_crossentropy, categorical_crossentropy

import warnings
warnings.filterwarnings("ignore")

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

IMG_SIZE = 256
NUM_CLASSES = 5
SEED = 26


# ## Create a training dataframe

# In[ ]:


train_dir = os.path.join('/kaggle/input','train_images/')
df = pd.read_csv(os.path.join('/kaggle/input', 'train.csv'))
df['path'] = df['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))
# df = df.drop(columns=['id_code'])
df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
df.head(10)


# In[ ]:


df_bin_train = df


# In[ ]:


# Uses a binary model. Changes values of 1-4 to 1
df_bin_train['diagnosis'] = df_bin_train['diagnosis'].map({0:0, 1: 1, 2: 1, 3: 1, 4: 1})


# In[ ]:


df_bin_train.head()


# In[ ]:


# train, test, split size of 
(train, valid) = train_test_split(df_bin_train, test_size=549, random_state=SEED)


# In[ ]:


# double check number of samples we have
print(len(train), len(valid))


# ## Create a testing dataframe for submission to Kaggle

# In[ ]:


test_dir = os.path.join('/kaggle/input','test_images/')
test_df = pd.read_csv(os.path.join('/kaggle/input', 'test.csv'))
test_df['path'] = test_df['id_code'].map(lambda x: os.path.join(test_dir,'{}.png'.format(x)))
# test_df = test_df.drop(columns=['id_code'])
# test_df = test_df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
test_df.head(10)


# In[ ]:


# cast 'diagnosis' as type string
df_bin_train['diagnosis'] = df_bin_train['diagnosis'].astype('str')
df_bin_train['diagnosis'].value_counts(normalize=True)


# In[ ]:


train['diagnosis'] = train['diagnosis'].astype('str')
valid['diagnosis'] = valid['diagnosis'].astype('str')


# In[ ]:


df_bin_train.head()


# # Display a few samples

# In[ ]:


def display_samples(df, columns=4, rows=3):
    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'path']
        image_id = df.loc[i,'diagnosis']
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig.add_subplot(rows, columns, i+1)
        plt.title(image_id)
        plt.imshow(img)
    
    plt.tight_layout()

display_samples(df)


# In[ ]:


total_count = df_bin_train['diagnosis'].sum()


# In[ ]:


df_bin_train['diagnosis'].unique()


# In[ ]:


# class_weight = class_weight.compute_class_weight('balanced',['0', '1', '2', '3', '4'],df['diagnosis'])
# class_weight = {0: 0.205913, 1: 0.130705, 2: 0.568465, 3: 0.052703, 4: 0.080557}
# class_weight = {0: -0.601, 1: -0.392, 2: -.833, 3: 0.164, 4: 1.662}


# ## Use ImageDataGenerator to add noise to pictures

# In[ ]:


datagen=ImageDataGenerator(featurewise_center=True,
                           featurewise_std_normalization=True,
                           rotation_range=20,
                           width_shift_range=0.1,
                           height_shift_range=0.1,
                           shear_range=16,
                           zoom_range=[0.9, 1.1],
                           fill_mode="constant",
                           cval=255,
                           horizontal_flip=True,
                           vertical_flip=True,
                           rescale=1./255.)


# In[ ]:


# datagen2 does not rescale values
# this is for better validation results
datagen2 = datagen=ImageDataGenerator(featurewise_center=True,
                           featurewise_std_normalization=True,
                           rotation_range=20,
                           width_shift_range=0.1,
                           height_shift_range=0.1,
                           shear_range=16,
                           zoom_range=[0.9, 1.1],
                           fill_mode="constant",
                           cval=255,
                           horizontal_flip=True,
                           vertical_flip=True)


# In[ ]:


# seq.augment_image(datagen)


# ## Use flow_frow_dataframe to create train_generator and test_generator

# In[ ]:


train_generator = datagen.flow_from_dataframe(
                                            dataframe=train,
                                            directory=None,
                                            x_col="path",
                                            y_col="diagnosis",
#                                             subset="training",
                                            batch_size=32,
                                            seed=SEED,
                                            shuffle=True,
                                            class_mode="binary",
                                            interpolation="bilinear",
                                            target_size=(IMG_SIZE, IMG_SIZE))
valid_generator = datagen2.flow_from_dataframe(
                                            dataframe=valid,
                                            directory=None,
                                            x_col="path",
                                            y_col="diagnosis",
#                                             subset="validation",
                                            batch_size=1,
                                            seed=SEED,
                                            shuffle=False,
                                            class_mode="binary",
                                            interpolation="bilinear",
                                            target_size=(IMG_SIZE, IMG_SIZE))
test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
                                            dataframe=test_df,
                                            directory=None,
                                            x_col="path",
                                            y_col=None,
                                            batch_size=32,
                                            seed=SEED,
                                            shuffle=False,
                                            class_mode=None,
                                            target_size=(IMG_SIZE, IMG_SIZE))


# In[ ]:


# change to_categorical
y_test = np_utils.to_categorical(valid_generator.classes, 2)


# In[ ]:


y_test


# In[ ]:


# verify our shapes
print(len(valid_generator.classes), len(y_test))


# In[ ]:


# method to define kaggle quadratic weighted kappa score
# only used on non binary model

from keras.callbacks import Callback
class QWKEvaluation(Callback):
    def __init__(self, validation_data=(), batch_size=64, interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.batch_size = batch_size
        self.valid_generator, self.y_val = validation_data
        self.history = []

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict_generator(generator=self.valid_generator,
#                                                   steps=np.ceil(float(len(self.y_val)) / float(self.batch_size)),
                                                  steps=18,
                                                  workers=1, use_multiprocessing=False,
                                                  verbose=1)
            def flatten(y):
                return np.argmax(y, axis=1).reshape(-1)
            try:
                score = cohen_kappa_score(self.y_val,
                                          flatten(y_pred),
                                          labels=[0,1,2,3,4],
                                          weights='quadratic')
                
                print("\n epoch: %d - QWK_score: %.6f \n" % (epoch+1, score))
                self.history.append(score)
                if score >= max(self.history):
                    print('saving checkpoint: ', score)
                    self.model.save('../working/densenet_bestqwk.h5')
            except:
                pass

qwk = QWKEvaluation(validation_data=(valid_generator, valid_generator.classes), batch_size=64, interval=1)


# In[ ]:


# define kappa_loss for kaggle
# only used on non-binary model
# reference link: https://www.kaggle.com/christofhenkel/weighted-kappa-loss-for-keras-tensorflow
def kappa_loss(y_true, y_pred, y_pow=2, eps=1e-12, N=5, bsize=32, name='kappa'):
    """A continuous differentiable approximation of discrete kappa loss.
        Args:
            y_pred: 2D tensor or array, [batch_size, num_classes]
            y_true: 2D tensor or array,[batch_size, num_classes]
            y_pow: int,  e.g. y_pow=2
            N: typically num_classes of the model
            bsize: batch_size of the training or validation ops
            eps: a float, prevents divide by zero
            name: Optional scope/name for op_scope.
        Returns:
            A tensor with the kappa loss."""

    with tf.name_scope(name):
        y_true = tf.to_float(y_true)
        repeat_op = tf.to_float(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]))
        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
        weights = repeat_op_sq / tf.to_float((N - 1) ** 2)
    
        pred_ = y_pred ** y_pow
        try:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
        except Exception:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))
    
        hist_rater_a = tf.reduce_sum(pred_norm, 0)
        hist_rater_b = tf.reduce_sum(y_true, 0)
    
        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)
    
        nom = tf.reduce_sum(weights * conf_mat)
        denom = tf.reduce_sum(weights * tf.matmul(
            tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /
                              tf.to_float(bsize))
    
        return nom*0.5 / (denom + eps) + categorical_crossentropy(y_true, y_pred)*0.5


# ## Create callbacks list

# In[ ]:


# callbacks list

early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=9)
checkpoint = ModelCheckpoint('../working/densenet_.h5', monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, 
                                   verbose=1, mode='auto', epsilon=0.0001)
csv_logger = CSVLogger(filename='../working/training_log.csv',
                       separator=',',
                       append=True)


# In[ ]:


# define metrics for model evaluation
def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[ ]:


# get the step sizes

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
STEP_SIZE_TEST = test_generator.n // test_generator.batch_size


# ## Add convolutions

# In[ ]:


# add convolutions

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
# model.add(Dropout(0.25))
model.add(layers.MaxPooling2D(2, 2))
model.add(Dropout(0.25))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))
# model.add(Dropout(0.25))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(Dropout(0.4))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(Dropout(0.4))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(Dropout(0.4))
model.add(layers.BatchNormalization())

model.summary()


# ## Add Dense layer and Output layer

# In[ ]:


# add dense layer

model.add(layers.Flatten()) #flatten 3D outputs to 1D
model.add(layers.Dense(128, activation='relu'))
model.add(Dropout(rate=0.3))
# model.add(layers.Dense(64, activation='relu'))
# model.add(Dropout(rate=0.5))
# model.add(layers.Dense(32, activation='relu'))
# model.add(Dropout(rate=0.5))
model.add(layers.Dense(1, activation='sigmoid')) # 1 output
model.summary()


# ## Compile the model

# In[ ]:


model.compile(optimizer='adam',
              loss = 'binary_crossentropy',
#               loss='categorical_crossentropy',
#               loss=kappa_loss,
              metrics=["acc"])


# In[ ]:


# callbacks_list = [checkpoint, csv_logger, reduceLROnPlat, early]
callbacks_list = [checkpoint, csv_logger, reduceLROnPlat, early, qwk]


# ## Fit the model

# In[ ]:


# fit the model
# commented out code for small trial set
history = model.fit_generator(generator=train_generator,
#                               steps_per_epoch=2,
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=valid_generator,
#                               validation_steps=2,
                              validation_steps=STEP_SIZE_VALID,
                              class_weight='balanced',
                              epochs=30,
                              verbose=1,
                              callbacks=callbacks_list
)


# In[ ]:


len(valid_generator.filenames)


# ## Create predictions

# In[ ]:


prediction = model.predict_generator(valid_generator, steps=549, verbose=1)


# In[ ]:


# create predicted class for each prediction
pred = []
for i in prediction:
    for j in i:
        if j > 0.5:
            pred.append(1)
        else:
            pred.append(0)


# In[ ]:


pred


# In[ ]:


pred = np.roll(pred, -1)


# In[ ]:


pred = np.roll(pred, 1)


# In[ ]:


# use this code for non biary model

# predictions = model.predict_generator(valid_generator, steps=len(valid_generator), verbose=1)        
# predictions = np.argmax(predictions, axis=-1) #multiple categories

# label_map = (train_generator.class_indices)
# label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
# predictions = [label_map[k] for k in predictions]


# In[ ]:


# predictions


# ## Evaluate the Model

# In[ ]:


# get the metrics names
model.metrics_names


# In[ ]:


# evaluate the model
model.evaluate_generator(generator=valid_generator,
                         steps=STEP_SIZE_TEST,
                         verbose=1)


# In[ ]:


import matplotlib.pyplot as plt

accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# ## Confusion Matrix

# In[ ]:


print('Confusion Matrix')
print(confusion_matrix(valid_generator.classes, pred))
print('Classification Report')
# target_names = ['0', '1', '2', '3', '4']
target_names = ['0', '1']
print(classification_report(valid_generator.classes, pred, target_names=target_names))


# In[ ]:


valid_generator.classes


# In[ ]:




