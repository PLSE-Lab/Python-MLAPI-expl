#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The purpose of this kernel is to show how to use the class ImageDataGenerator while loading the images from a folder.
# 
# I heavily inspired from this kernel:
# https://www.kaggle.com/kaushal2896/bengali-graphemes-starter-eda-multi-output-cnn
# 
# We will use the images dataset provided by iafoss:
# https://www.kaggle.com/iafoss/image-preprocessing-128x128

# ## Load packages

# In[ ]:


import os
import warnings
import cv2

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Recall
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    BatchNormalization,
    Dropout,
    LeakyReLU,
)
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence, plot_model

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score

warnings.filterwarnings("ignore")


# In[ ]:


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
stats = (0.0692, 0.2051)


# ## Basic data exploration

# In[ ]:


DATA_PATH = '/kaggle/input/bengaliai-cv19/'
IMG_PATH = '/kaggle/input/grapheme-imgs-128x128/'

train = pd.read_csv(f'{DATA_PATH}train.csv')
train['filename'] = train['image_id'] + '.png'  # This column will be used by the ImageDataGenerator
test = pd.read_csv(f'{DATA_PATH}test.csv')
class_map = pd.read_csv(f'{DATA_PATH}class_map.csv')
sample_submission = pd.read_csv(f'{DATA_PATH}sample_submission.csv')


# In[ ]:


print(train.shape, test.shape, sample_submission.shape)
train.head()


# In[ ]:


test.head()


# In[ ]:


train.info()


# ## Image Loaders
# 
# We will use two classes:
# 
# - **MultiOutputDataGenerator**: based on Keras ImageDataGenerator. Used for data augmentation.
# - **ImageGenerator**: based on Keras Sequence. Used for loading and preprocessing images in batches.

# In[ ]:


HEIGHT = 137
WIDTH = 236
IMG_SIZE = 128
N_CHANNELS = 1

BATCH_SIZE = 128
input_shape = (IMG_SIZE, IMG_SIZE, N_CHANNELS)


# In[ ]:


class MultiOutputDataGenerator(ImageDataGenerator):
    def flow_from_dataframe(
        self,
        dataframe,
        directory=None,
        x_col='filename',
        y_col='class',
        weight_col=None,
        target_size=(256, 256),
        color_mode='rgb',
        classes=None,
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=None,
        save_to_dir=None,
        save_prefix='',
        save_format='png',
        subset=None,
        interpolation='nearest',
        validate_filenames=True,
        **kwargs
    ):

        for flow_x, flow_y in super().flow_from_dataframe(
            dataframe,
            directory=directory,
            x_col=x_col,
            y_col=y_col,
            weight_col=weight_col,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset,
            interpolation=interpolation,
            validate_filenames=validate_filenames
        ):
            # The flow_y will have shape 128 * 3. We want it to be a list of 3 numpy arrays
            # with the following shapes [128 * 168, 128 * 11, 128 * 7]
            Y_root = kwargs.get('le_root').transform(flow_y[:,0])
            Y_vowel = kwargs.get('le_vowel').transform(flow_y[:,1])
            Y_consonant = kwargs.get('le_consonant').transform(flow_y[:,2])

            yield flow_x, [Y_root, Y_vowel, Y_consonant]


# In[ ]:


class ImageGenerator(Sequence):
    def __init__(self, data, batch_size, dim, shuffle=True, **kwargs):
        self.data = data
        self.list_ids = data.index.values
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.le_root = kwargs.get('le_root')
        self.le_vowel = kwargs.get('le_vowel')
        self.le_consonant = kwargs.get('le_consonant')
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.data) // self.batch_size)

    def __getitem__(self, index):
        batch_ids = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        valid_ids  = [self.list_ids[i] for i in batch_ids]

        X = np.empty((self.batch_size, *self.dim, 1))
        Y_root = self.le_root.transform(self.data.loc[valid_ids, 'grapheme_root'].values)
        Y_vowel = self.le_vowel.transform(self.data.loc[valid_ids, 'vowel_diacritic'].values)
        Y_consonant = self.le_consonant.transform(self.data.loc[valid_ids, 'consonant_diacritic'].values)
        
        for i, k in enumerate(valid_ids):
            img_path = f'{IMG_PATH}{self.data["image_id"][k]}.png'
            img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
            img = img[:, :, np.newaxis]
            X[i, :, :, :] = img

        return X, [Y_root, Y_vowel, Y_consonant]

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)


# ## Build Model
# 
# We use a miniVGG architecture.

# In[ ]:


input_shape = (IMG_SIZE, IMG_SIZE, N_CHANNELS)

# TODO: replace this with a better model
def build_model():
    inputs = Input(shape=input_shape)

    chan_dim = -1
    # first CONV => RELU => CONV => RELU => POOL layer set
    model = Conv2D(
        32, (3, 3), padding="same", input_shape=input_shape, activation="relu"
    )(inputs)
    model = BatchNormalization(axis=chan_dim)(model)
    model = Conv2D(32, (3, 3), padding="same", activation="relu")(model)
    model = BatchNormalization(axis=chan_dim)(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(0.25)(model)

    # second CONV => RELU => CONV => RELU => POOL layer set
    model = Conv2D(64, (3, 3), padding="same", activation="relu")(model)
    model = BatchNormalization(axis=chan_dim)(model)
    model = Conv2D(64, (3, 3), padding="same", activation="relu")(model)
    model = BatchNormalization(axis=chan_dim)(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(0.25)(model)

    # first (and only) set of FC => RELU layers
    model = Flatten()(model)
    model = Dense(512, activation="relu")(model)
    model = BatchNormalization()(model)
    model = Dropout(0.5)(model)

    # softmax classifier
    head_root = Dense(168, activation="softmax")(model)
    head_vowel = Dense(11, activation="softmax")(model)
    head_consonant = Dense(7, activation="softmax")(model)

    model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])

    return model


# In[ ]:


model = build_model()
model.summary()


# In[ ]:


plot_model(model, to_file='model.png')


# In[ ]:


opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model.compile(
    optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy", Recall()]
)


# In[ ]:


le_root = LabelBinarizer()
_ = le_root.fit_transform(train['grapheme_root'].values)

le_vowel = LabelBinarizer()
_ = le_vowel.fit_transform(train['vowel_diacritic'].values)

le_consonant = LabelBinarizer()
_ = le_consonant.fit_transform(train['consonant_diacritic'].values)


# We will use the **MultiOutputDataGenerator** for the training set and the **ImageGenerator** for the validation set.

# In[ ]:


trainX, valX = train_test_split(train, test_size=0.15, random_state=SEED)
train_generator = MultiOutputDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range = 0.15, # Randomly zoom image 
    width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False   # randomly flip images,
)
val_generator = ImageGenerator(
    data=valX,
    batch_size=BATCH_SIZE,
    dim=(IMG_SIZE, IMG_SIZE),
    **{'le_root': le_root, 'le_vowel': le_vowel, 'le_consonant': le_consonant}
)


# ## Training the model

# In[ ]:


get_ipython().run_cell_magic('time', '', "# TODO: run this with more epochs\nEPOCHS = 10\nhistory = model.fit_generator(\n    train_generator.flow_from_dataframe(\n        dataframe=train,\n        directory=IMG_PATH,\n        x_col='filename',\n        y_col=['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'],\n        class_mode='other',\n        batch_size=BATCH_SIZE,\n        target_size=(IMG_SIZE, IMG_SIZE), # Default value is 256 x 256\n        color_mode='grayscale',\n        shuffle=False,\n        **{'le_root': le_root, 'le_vowel': le_vowel, 'le_consonant': le_consonant}\n    ),\n    steps_per_epoch=int(trainX.shape[0] / BATCH_SIZE),\n    validation_data=val_generator,\n    validation_steps=int(valX.shape[0] / BATCH_SIZE),\n    epochs=EPOCHS\n)")


# ## Evaluate the model

# In[ ]:


root_score = np.mean(history.history['val_dense_1_recall'])
vowel_score = np.mean(history.history['val_dense_2_recall'])
consonant_score = np.mean(history.history['val_dense_3_recall'])
print(root_score, vowel_score, consonant_score, 0.5 * root_score + 0.25 * vowel_score + 0.25 * consonant_score)


# In[ ]:


def plot_loss(his, prefix, epoch, title):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, epoch), his.history[f'{prefix}_loss'], label='train_loss')
    plt.plot(np.arange(0, epoch), his.history[f'val_{prefix}_loss'], label='val_loss')
    plt.title(title)
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

def plot_acc(his, prefix, epoch, title):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, epoch), his.history[f'{prefix}_accuracy'], label='train_acc')
    plt.plot(np.arange(0, epoch), his.history[f'val_{prefix}_accuracy'], label='val_accuracy')
    plt.title(title)
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.show()

def plot_results():
    m = {'dense_1': 'root', 'dense_2': 'vowel', 'dense_3': 'consonant'}
    for ol in ['dense_1', 'dense_2', 'dense_3']:
        plot_loss(history, ol, EPOCHS, f'Training on: {m[ol]}')
        plot_acc(history, ol, EPOCHS, f'Training on: {m[ol]}')


# In[ ]:


plot_results()


# ## TODOS
# 
# - Replace miniVGG net with a better model
# - Use more epochs in your training

# In[ ]:




