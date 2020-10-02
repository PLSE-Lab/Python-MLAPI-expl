#!/usr/bin/env python
# coding: utf-8

# # PANDA: simple keras baseline

# The idea to use tiles of original image was taken from <a href="https://www.kaggle.com/iafoss/panda-16x128x128-tiles">this great notebook</a>. 
# 
# The rest of current notebook is rather simple - just glue all tiles to one image and put in through keras based model with <a href="https://github.com/qubvel/efficientnet">EfficientNet</a> as backbone.

# In[ ]:


get_ipython().run_cell_magic('time', '', '!pip install ../input/efficientnet/efficientnet-1.1.0/ -f ./ --no-index')


# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import os
import cv2
import numpy as np
import pandas as pd 
import json
import skimage.io
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import efficientnet.tfkeras as efn
import albumentations as albu
print('tensorflow version:', tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print('no gpus')


# In[ ]:


DATA_PATH = '../input/prostate-cancer-grade-assessment'
MODELS_PATH = '.'
IMG_SIZE = 64
SEQ_LEN = 25
BATCH_SIZE = 16
MDL_VERSION = 'v0'
SEED = 80


# ## Data prepare

# Data generator to feed neural network takes image, cut it to tiles and produces image that made of tiles:

# In[ ]:


def get_axis_max_min(array, axis=0):
    one_axis = list((array != 255).sum(axis=tuple([x for x in (0, 1, 2) if x != axis])))
    axis_min = next((i for i, x in enumerate(one_axis) if x), 0)
    axis_max = len(one_axis) - next((i for i, x in enumerate(one_axis[::-1]) if x), 0)
    return axis_min, axis_max


# In[ ]:


class DataGenPanda(Sequence):
    def __init__(self, imgs_path, df, batch_size=32, 
                 mode='fit', shuffle=False, aug=None, 
                 seq_len=12, img_size=128, n_classes=6):
        self.imgs_path = imgs_path
        self.df = df
        self.shuffle = shuffle
        self.mode = mode
        self.aug = aug
        self.batch_size = batch_size
        self.img_size = img_size
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.side = int(seq_len ** .5)
        self.on_epoch_end()
    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    def __getitem__(self, index):
        X = np.zeros((self.batch_size, self.side * self.img_size, self.side * self.img_size, 3), dtype=np.float32)
        imgs_batch = self.df[index * self.batch_size : (index + 1) * self.batch_size]['image_id'].values
        for i, img_name in enumerate(imgs_batch):
            img_path = '{}/{}.tiff'.format(self.imgs_path, img_name)
            img_patches = self.get_patches(img_path)
            X[i, ] = self.glue_to_one(img_patches)
        if self.mode == 'fit':
            y = np.zeros((self.batch_size, self.n_classes), dtype=np.float32)
            lbls_batch = self.df[index * self.batch_size : (index + 1) * self.batch_size]['isup_grade'].values
            for i in range(self.batch_size):
                y[i, lbls_batch[i]] = 1
            return X, y
        elif self.mode == 'predict':
            return X
        else:
            raise AttributeError('mode parameter error')
    def get_patches(self, img_path):
        num_patches = self.seq_len
        p_size = self.img_size
        img = skimage.io.MultiImage(img_path)[-1]
        a0min, a0max = get_axis_max_min(img, axis=0)
        a1min, a1max = get_axis_max_min(img, axis=1)
        img = img[a0min:a0max, a1min:a1max, :].astype(np.float32) / 255
        if self.aug:
            img = self.aug(image=img)['image']
        pad0, pad1 = (p_size - img.shape[0] % p_size) % p_size, (p_size - img.shape[1] % p_size) % p_size
        img = np.pad(
            img,
            [
                [pad0 // 2, pad0 - pad0 // 2], 
                [pad1 // 2, pad1 - pad1 // 2], 
                [0, 0]
            ],
            constant_values=1
        )
        img = img.reshape(img.shape[0] // p_size, p_size, img.shape[1] // p_size, p_size, 3)
        img = img.transpose(0, 2, 1, 3, 4).reshape(-1, p_size, p_size, 3)
        if len(img) < num_patches:
            img = np.pad(
                img, 
                [
                    [0, num_patches - len(img)],
                    [0, 0],
                    [0, 0],
                    [0, 0]
                ],
                constant_values=1
            )
        idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:num_patches]
        return np.array(img[idxs])
    def glue_to_one(self, imgs_seq):
        img_glue = np.zeros((self.img_size * self.side, self.img_size * self.side, 3), dtype=np.float32)
        for i, ptch in enumerate(imgs_seq):
            x = i // self.side
            y = i % self.side
            img_glue[x * self.img_size : (x + 1) * self.img_size, 
                     y * self.img_size : (y + 1) * self.img_size, :] = ptch
        return img_glue


# Load train metadata, train-test split with classes balance:

# In[ ]:


train = pd.read_csv('{}/train.csv'.format(DATA_PATH))
print('train: ', train.shape, '| unique ids:', sum(train['isup_grade'].value_counts()))
X_train, X_val = train_test_split(train, test_size=.2, stratify=train['isup_grade'], random_state=SEED)
lbl_value_counts = X_train['isup_grade'].value_counts()
class_weights = {i: max(lbl_value_counts) / v for i, v in lbl_value_counts.items()}
print('classes weigths:', class_weights)


# In[ ]:


aug = albu.Compose(
    [
        albu.OneOf([albu.RandomBrightness(limit=.15), albu.RandomContrast(limit=.3), albu.RandomGamma()], p=.25),
        albu.HorizontalFlip(p=.25),
        albu.VerticalFlip(p=.25),
        albu.ShiftScaleRotate(shift_limit=.1, scale_limit=.1, rotate_limit=20, p=.25)
    ]
)
train_datagen = DataGenPanda(
    imgs_path='{}/train_images'.format(DATA_PATH), 
    df=X_train, 
    batch_size=BATCH_SIZE,
    mode='fit', 
    shuffle=True, 
    aug=aug, 
    seq_len=SEQ_LEN, 
    img_size=IMG_SIZE, 
    n_classes=6
)
val_datagen = DataGenPanda(
    imgs_path='{}/train_images'.format(DATA_PATH), 
    df=X_val, 
    batch_size=BATCH_SIZE,
    mode='fit', 
    shuffle=False, 
    aug=None, 
    seq_len=SEQ_LEN, 
    img_size=IMG_SIZE, 
    n_classes=6
)


# Let's look at result that data generator produces, just to see it is normal as train data:

# In[ ]:


Xt, yt = train_datagen.__getitem__(0)
print('test X: ', Xt.shape)
print('test y: ', yt.shape)
fig, axes = plt.subplots(figsize=(18, 6), ncols=BATCH_SIZE)
for j in range(BATCH_SIZE):
    axes[j].imshow(Xt[j])
    axes[j].axis('off')
    axes[j].set_title('label {}'.format(np.argmax(yt[j, ])))
plt.show()


# ## Train model

# Our network based on EfficientNetB3:

# In[ ]:


bottleneck = efn.EfficientNetB3(
    input_shape=(int(SEQ_LEN ** .5) * IMG_SIZE, int(SEQ_LEN ** .5) * IMG_SIZE, 3),
    weights='../input/effnetweights/efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5', 
    include_top=False, 
    pooling='avg'
)
bottleneck = Model(inputs=bottleneck.inputs, outputs=bottleneck.layers[-2].output)
model = Sequential()
model.add(bottleneck)
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dropout(.25))
model.add(Dense(512, activation='elu'))
model.add(BatchNormalization())
model.add(Dropout(.25))
model.add(Dense(6, activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


#def qw_kappa_score(y_true, y_pred):     
#    y_true=tf.math.argmax(y_true, axis=1)
#    y_pred=tf.math.argmax(y_pred, axis=1)
#    def sklearn_qwk(y_true, y_pred) -> np.float64:
#        return cohen_kappa_score(y_true, y_pred, weights='quadratic')
#    return tf.compat.v1.py_func(sklearn_qwk, (y_true, y_pred), tf.double)


# In[ ]:


import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

import keras.backend as K
import tensorflow as tf


def kappa_keras(y_true, y_pred):

    y_true = K.cast(K.argmax(y_true, axis=-1), dtype='int32')
    y_pred = K.cast(K.argmax(y_pred, axis=-1), dtype='int32')
    #print(y_true)
    #print(y_pred)
    # Figure out normalized expected values
    min_rating = K.minimum(K.min(y_true), K.min(y_pred))
    max_rating = K.maximum(K.max(y_true), K.max(y_pred))

    # shift the values so that the lowest value is 0
    # (to support scales that include negative values)
    y_true = K.map_fn(lambda y: y - min_rating, y_true, dtype='int32')
    y_pred = K.map_fn(lambda y: y - min_rating, y_pred, dtype='int32')

    # Build the observed/confusion matrix
    num_ratings = max_rating - min_rating + 1
    observed = tf.math.confusion_matrix(y_true, y_pred,
                                num_classes=num_ratings)
    num_scored_items = K.shape(y_true)[0]

    weights = K.expand_dims(K.arange(num_ratings), axis=-1) - K.expand_dims(K.arange(num_ratings), axis=0)
    weights = K.cast(K.pow(weights, 2), dtype='float64')

    hist_true = tf.math.bincount(y_true, minlength=num_ratings)
    hist_true = hist_true[:num_ratings] / num_scored_items
    hist_pred = tf.math.bincount(y_pred, minlength=num_ratings)
    hist_pred = hist_pred[:num_ratings] / num_scored_items
    expected = K.dot(K.expand_dims(hist_true, axis=-1), K.expand_dims(hist_pred, axis=0))

    # Normalize observed array
    observed = observed / num_scored_items

    # If all weights are zero, that means no disagreements matter.
    score = tf.where(K.any(K.not_equal(weights, 0)), 
                     K.sum(weights * observed) / K.sum(weights * expected), 
                     0)
    
    return 1. - score

if __name__ == '__main__':
    y_true = np.array([2, 0, 2, 2, 0, 1])
    y_pred = np.array([0, 0, 2, 2, 0, 2])
    # Testing Keras implementation of QWK
    
    # Calculating QWK score with scikit-learn
   
    skl_score = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    # Keras implementation of QWK work with one hot encoding labels and predictions (also it works with softmax probabilities)
    # Converting arrays to one hot encoded representation
    shape = (y_true.shape[0], np.maximum(y_true.max(), y_pred.max()) + 1)

    y_true_ohe = np.zeros(shape)
    y_true_ohe[np.arange(shape[0]), y_true] = 1

    y_pred_ohe = np.zeros(shape)
    y_pred_ohe[np.arange(shape[0]), y_pred] = 1
    
    # Calculating QWK score with Keras
    with tf.compat.v1.Session() as sess:
        keras_score = kappa_keras(y_true_ohe, y_pred_ohe).eval()
    
    #print('Scikit-learn score: {:.03}, Keras score: {:.03}'.format(skl_score, keras_score))
    


# In[ ]:


model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(lr=1e-3),
    metrics=['categorical_accuracy', kappa_keras]
)


# Train for only 20 epochs for demo:

# In[ ]:





# In[ ]:


get_ipython().run_cell_magic('time', '', "model_file = '{}/model_{}.h5'.format(MODELS_PATH, MDL_VERSION)\nif False:\n    model = load_model(model_file)\n    print('model loaded')\nelse:\n    print('train from scratch')\nEPOCHS = 20\nearlystopper = EarlyStopping(\n    monitor='val_loss', \n    patience=10, \n    verbose=1,\n    mode='min'\n)\nmodelsaver = ModelCheckpoint(\n    model_file, \n    monitor='val_loss', \n    verbose=1, \n    save_best_only=True,\n    mode='min'\n)\nlrreducer = ReduceLROnPlateau(\n    monitor='val_loss',\n    factor=.1,\n    patience=5,\n    verbose=1,\n    min_lr=1e-7\n)\nhistory = model.fit_generator(\n    train_datagen,\n    validation_data=val_datagen,\n    class_weight=class_weights,\n    callbacks=[earlystopper, modelsaver, lrreducer],\n    epochs=EPOCHS,\n    verbose=1\n)")


# In[ ]:


history_file = '{}/history_{}.txt'.format(MODELS_PATH, MDL_VERSION)
dict_to_save = {}
for k, v in history.history.items():
    dict_to_save.update({k: [np.format_float_positional(x) for x in history.history[k]]})
with open(history_file, 'w') as file:
    json.dump(dict_to_save, file)
ep_max = EPOCHS
plt.plot(history.history['loss'][:ep_max], label='loss')
plt.plot(history.history['val_loss'][:ep_max], label='val_loss')
plt.legend()
plt.show()
plt.plot(history.history['categorical_accuracy'][:ep_max], label='cat. accuracy')
plt.plot(history.history['val_categorical_accuracy'][:ep_max], label='val_accuracy')
plt.plot(history.history['kappa_keras'][:ep_max], label='kappa_keras')
plt.plot(history.history['val_kappa_keras'][:ep_max], label='val_kappa_keras')
plt.legend()
plt.show()


# ## Inference

# In[ ]:


test = pd.read_csv('{}/test.csv'.format(DATA_PATH))
preds = [[0] * 6] * len(test)
if os.path.exists('../input/prostate-cancer-grade-assessment/test_images'):
    subm_datagen = DataGenPanda(
        imgs_path='{}/test_images'.format(DATA_PATH), 
        df=test,
        batch_size=1,
        mode='predict', 
        shuffle=False, 
        aug=None, 
        seq_len=SEQ_LEN, 
        img_size=IMG_SIZE, 
        n_classes=6
    )
    preds = model.predict_generator(subm_datagen)
    print('predictions done, total:', len(preds))
else:
    print('submission not found')
test['isup_grade'] = np.argmax(preds, axis=1)
test.drop('data_provider', axis=1, inplace=True)
test.to_csv('submission.csv', index=False)
print('submission saved')


# In[ ]:


#Original Author: https://www.kaggle.com/vgarshin

