#!/usr/bin/env python
# coding: utf-8

# # PANDA: keras with timedistributed layer

# The idea to use tiles of original image was taken from <a href="https://www.kaggle.com/iafoss/panda-16x128x128-tiles">this great notebook</a>. 
# 
# The rest of current notebook is rather simple - put sequence of tiles to CNN (<a href="https://github.com/qubvel/efficientnet">EfficientNet</a> as backbone) with wrapper layer (timedistributed) and classify result.

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
from sklearn.metrics import cohen_kappa_score
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


from tensorflow.keras.utils import get_custom_objects
class Mish(Activation):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''
    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'
def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))
get_custom_objects().update({'Mish': Mish(mish)})


# In[ ]:


DATA_PATH = '../input/prostate-cancer-grade-assessment'
MODELS_PATH = '.'
# Use half-sized images for EfficientNet input
# Inputs for EfficientNet:
# B0: 224x224x3 
# B1: 240x240x3
# B2: 260x260x3
# B3: 300x300x3
# B4: 380x380x3
# B5: 456x456x3
IMG_SIZE = 120 # half-sized B1
SEQ_LEN = 10
BATCH_SIZE = 20
MDL_VERSION = 'v2'
TIFF = -1
RESIZE = None
SEED = 80


# ## Data prepare

# Data generator to feed neural network takes image, cut it to tiles and produces sequence of tiles:

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
                 tiff=-1, resize=None,
                 seq_len=12, img_size=128, n_classes=6):
        self.imgs_path = imgs_path
        self.df = df
        self.shuffle = shuffle
        self.mode = mode
        self.aug = aug
        self.tiff = tiff
        self.resize = resize
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
        X = np.zeros((self.batch_size, self.seq_len, self.img_size, self.img_size, 3), dtype=np.float32)
        imgs_batch = self.df[index * self.batch_size : (index + 1) * self.batch_size]['image_id'].values
        for i, img_name in enumerate(imgs_batch):
            img_path = '{}/{}.tiff'.format(self.imgs_path, img_name)
            img_patches = self.get_patches(img_path)
            X[i, ] = img_patches
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
        img = skimage.io.MultiImage(img_path)[self.tiff]
        if self.resize:
            img = cv2.resize(img, (int(img.shape[1] / self.resize), int(img.shape[0] / self.resize)))
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
        albu.OneOf(
            [
                albu.RandomBrightness(limit=.15), 
                albu.RandomContrast(limit=.3), 
                albu.RandomGamma()
            ], 
            p=.3
        ),
        albu.HorizontalFlip(p=.3),
        albu.VerticalFlip(p=.3),
        albu.ShiftScaleRotate(shift_limit=.1, scale_limit=.1, rotate_limit=20, p=.3)
    ]
)
train_datagen = DataGenPanda(
    imgs_path='{}/train_images'.format(DATA_PATH), 
    df=X_train, 
    batch_size=BATCH_SIZE,
    mode='fit', 
    shuffle=True, 
    aug=aug, 
    tiff=TIFF,
    resize=RESIZE,
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
    tiff=TIFF,
    resize=RESIZE,
    seq_len=SEQ_LEN, 
    img_size=IMG_SIZE, 
    n_classes=6
)


# Let's look at result that data generator producers, just to see it is normal as train data:

# In[ ]:


Xt, yt = train_datagen.__getitem__(0)
print('test X: ', Xt.shape)
print('test y: ', yt.shape)
fig, axes = plt.subplots(figsize=(SEQ_LEN, BATCH_SIZE), nrows=BATCH_SIZE, ncols=SEQ_LEN)
for j in range(BATCH_SIZE):
    for i in range(SEQ_LEN):
        axes[j, i].imshow(Xt[j][i])
        axes[j, i].axis('off')
        axes[j, i].set_title(np.argmax(yt[j, ]))
plt.show()


# ## Train model

# Our network based on EfficientNet:

# In[ ]:


bottleneck = efn.EfficientNetB1(
    weights='../input/effnetweights/efficientnet-b1_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5', 
    include_top=False, 
    pooling='avg'
)
bottleneck = Model(inputs=bottleneck.inputs, outputs=bottleneck.layers[-2].output)
model = Sequential()
model.add(TimeDistributed(bottleneck, input_shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 3)))
model.add(TimeDistributed(BatchNormalization()))
model.add(GlobalMaxPooling3D())
model.add(BatchNormalization())
model.add(Dropout(.4))
model.add(Dense(512, activation='Mish'))
model.add(BatchNormalization())
model.add(Dropout(.4))
model.add(Dense(128, activation='Mish'))
model.add(BatchNormalization())
model.add(Dropout(.4))
model.add(Dense(6, activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


def qw_kappa_score(y_true, y_pred):
    y_true=tf.math.argmax(y_true, axis=1)
    y_pred=tf.math.argmax(y_pred, axis=1)
    def sklearn_qwk(y_true, y_pred) -> np.float64:
        return cohen_kappa_score(y_true, y_pred, weights='quadratic')
    return tf.compat.v1.py_func(sklearn_qwk, (y_true, y_pred), tf.double)
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(lr=1e-3),
    metrics=['categorical_accuracy', qw_kappa_score]
)


# Train for only 15 epochs for demo:

# In[ ]:


get_ipython().run_cell_magic('time', '', "model_file = '{}/model_{}.h5'.format(MODELS_PATH, MDL_VERSION)\nif False:\n    model = load_model(model_file)\n    print('model loaded')\nelse:\n    print('train from scratch')\nEPOCHS = 15\nearlystopper = EarlyStopping(\n    monitor='val_qw_kappa_score', \n    patience=10, \n    verbose=1,\n    mode='max'\n)\nmodelsaver = ModelCheckpoint(\n    model_file, \n    monitor='val_qw_kappa_score', \n    verbose=1, \n    save_best_only=True,\n    mode='max'\n)\nlrreducer = ReduceLROnPlateau(\n    monitor='val_qw_kappa_score',\n    factor=.1,\n    patience=5,\n    verbose=1,\n    min_lr=1e-7\n)\nhistory = model.fit_generator(\n    train_datagen,\n    validation_data=val_datagen,\n    class_weight=class_weights,\n    callbacks=[earlystopper, modelsaver, lrreducer],\n    epochs=EPOCHS,\n    verbose=1\n)")


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
plt.plot(history.history['categorical_accuracy'][:ep_max], label='cat acc')
plt.plot(history.history['val_categorical_accuracy'][:ep_max], label='val cat acc')
plt.legend()
plt.show()
plt.plot(history.history['qw_kappa_score'][:ep_max], label='qwk')
plt.plot(history.history['val_qw_kappa_score'][:ep_max], label='val qwk')
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
        tiff=TIFF,
        resize=RESIZE,
        seq_len=SEQ_LEN, 
        img_size=IMG_SIZE, 
        n_classes=6
    )
    preds = model.predict_generator(subm_datagen)
    print('preds done, total:', len(preds))
else:
    print('preds are zeros')
test['isup_grade'] = np.argmax(preds, axis=1)
test.drop('data_provider', axis=1, inplace=True)
test.to_csv('submission.csv', index=False)
print('submission saved')


# In[ ]:




