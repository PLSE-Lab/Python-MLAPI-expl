#!/usr/bin/env python
# coding: utf-8

# ## Import necessary libraries

# In[ ]:


# tensorflow
import tensorflow as tf

# Keras modules
import keras
from keras.callbacks import ModelCheckpoint
from keras import backend
# from tensorflow.keras.applications import DenseNet --> Discarded due to too much memory usage
from keras.applications import InceptionV3
from keras.applications import InceptionResNetV2
from keras.applications import MobileNet
from keras.applications import ResNet50
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import Xception
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.models import load_model
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import Sequence

# data processing modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# metrics functions
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

# plotting
import matplotlib.pyplot as plt

# image processing
import cv2
from PIL import Image
from imgaug import augmenters as iaa

# python support libraries
import os
import datetime
import pickle
from zipfile import ZipFile
from collections import Iterable
import warnings
import itertools
warnings.filterwarnings("ignore")


# ## Functions definitions 

# In[ ]:


def get_classes(x):
    """
    Transform the Target column from the train dataset into an array of classes found in the image
    to be later used as a target variable
    :param x: row value for column Target from apply function
    :return: array of length 27 with 0 and 1
    """
    return np.array([0 if str(i) not in str(x).split(' ') else 1 for i in range(28)])


def load_image(file_path, image_name, shape=(512, 512, 3)):
    """
    Load image contained within a zip file
    :param file_path:  (string) Path to image on disk
    :param image_name: (string) Image name contained in zip file
    :param shape:      (tuple) Shape of image to be outputed
    :return:           (array) 3D of RGBY divided by 255
    """
    # load images by channel
    channel_list = list()
    for c in ['red', 'green', 'blue', 'yellow']:
        img = cv2.imread(file_path + '/' + image_name + '_' + c + '.png', 0)
        img = cv2.resize(img, dsize=(shape[0], shape[1]), interpolation=cv2.INTER_CUBIC)
        img = np.array(img)
        channel_list.append(img)
    
    # stack pixels of image
    if shape[2] == 3:
        image = np.stack((
            channel_list[0]/2 + channel_list[3]/2, 
            channel_list[1]/2 + channel_list[3]/2, 
            channel_list[2]
        ),-1)
    else:
        image = np.stack(channel_list, -1)

    # normalize pixels range
    image = np.divide(image, 255)

    # return array with normalized colors
    return image


def f1(y_true, y_pred):
    """
    Calculate the f1 score given the true values and predictions
    :param y_true: (array) true value array
    :param y_pred: (array) predictions array
    :return: (float) f1 score
    """
    tp = backend.sum(backend.cast(y_true * y_pred, 'float'), axis=0)
    fp = backend.sum(backend.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = backend.sum(backend.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + backend.epsilon())
    r = tp / (tp + fn + backend.epsilon())

    f1 = 2 * p * r / (p + r + backend.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return backend.mean(f1)


def focal_loss(gamma=2., alpha=.25):
    """
    Function to use the focal loss function
    source: https://github.com/mkocabas/focal-loss-keras
    :param gamma: (float) gamma value
    :param alpha: (float) alpha value
    :return: (function) parameter adjusted focal loss function
    """
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -keras.backend.sum(alpha * keras.backend.pow(1. - pt_1, gamma) * keras.backend.log(pt_1))-keras.backend.sum((1-alpha) * keras.backend.pow( pt_0, gamma) * keras.backend.log(1. - pt_0))
    return focal_loss_fixed


def get_weighted_loss(weights):
    """
    Function to use the weighted loss function
    source: https://stackoverflow.com/questions/48485870/multi-label-classification-with-class-weights-in-keras
    :param weights: (array) weights dictionary
    :return: (function) parameter adjusted weighted loss function
    """
    def weighted_loss(y_true, y_pred):
        return keras.backend.mean((weights[:, 0] ** (1-y_true)) * (weights[:,1] ** (y_true)) * keras.backend.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss


def augment(image):
    """
    Apply transformations to images
    source: https://www.kaggle.com/rejpalcz/cnn-128x128x4-keras-from-scratch-lb-0-328
    :param image:
    :return:
    """
    augment_img = iaa.Sequential([
        iaa.OneOf([
            # flip the image
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),

            # random crops
            iaa.Crop(percent=(0, 0.1)),

            # Strengthen or weaken the contrast in each image
            iaa.ContrastNormalization((0.75, 1.5)),

            # Add gaussian noise
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

            # Make some images brighter and some darker
            iaa.Multiply((0.8, 1.2), per_channel=0.2),

            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-180, 180),
                shear=(-8, 8)
            )
        ])], random_order=True)

    image_aug = augment_img.augment_image(image)
    return image_aug


def show_history(history):
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('f1')
    ax[1].plot(history.epoch, history.history["f1"], label="Train f1")
    ax[1].plot(history.epoch, history.history["val_f1"], label="Validation f1")
    ax[2].set_title('acc')
    ax[2].plot(history.epoch, history.history["acc"], label="Train acc")
    ax[2].plot(history.epoch, history.history["val_acc"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()


def sequencial_model(input_shape, n_out):
    # Initialising the CNN
    model = Sequential()

    # ##### LAYER 1
    # model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # ##### LAYER 2
    # model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # ##### FLATTENING
    model.add(Flatten())

    # ##### ANN
    model.add(Dense(activation='relu', units=1024))
    model.add(Dense(activation='relu', units=128))
    model.add(Dense(activation='sigmoid', units=n_out))

    return model


def vgg_model(input_shape, n_out):
    model = VGG19(
        include_top=False, weights='imagenet', input_shape=input_shape
    )

    input_tensor = Input(shape=input_shape)
    bn = BatchNormalization()(input_tensor)
    x = model(bn)
    x = Conv2D(128, kernel_size=(1, 1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_out, activation='sigmoid')(x)
    return Model(input_tensor, output)


def inception_res_net_model(input_shape, n_out):    
    pretrain_model = InceptionResNetV2(
        include_top=False, 
        weights='imagenet', 
        input_shape=input_shape
    )    
    
    input_tensor = Input(shape=input_shape)
    bn = BatchNormalization()(input_tensor)
    x = pretrain_model(bn)
    x = Conv2D(128, kernel_size=(1,1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    
    return model


def res_net_50(input_shape, n_out):    
    pretrain_model = ResNet50(
        include_top=False, 
        weights='imagenet', 
        input_shape=input_shape
    )    
    
    input_tensor = Input(shape=input_shape)
    bn = BatchNormalization()(input_tensor)
    x = pretrain_model(bn)
    x = Conv2D(128, kernel_size=(1, 1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    
    return model


# ## Classes definition

# In[ ]:


class StratifiedMultiLabelKFold(object):
    """
    Create a stratified k fold object capable of splitting the dataset in even parts according to 
    the multi labels contained within it
    """
    def __init__(self, n_splits, shuffle=False, random_state=None):
        """
        :param n_splits: (int) Number of folds. Must be at least 2
        :param shuffle:  (bool) Whether to shuffle each stratification of the data before splitting into batches
        :param random_state: (int, obj) If int, random_state is the seed used by the random number generator; 
                                        If RandomState instance, random_state is the random number generator; 
                                        If None, the random number generator is the RandomState instance used by np.random. 
                                        Used when shuffle == True
        """
        assert type(n_splits) == int
        assert n_splits >= 2
        assert type(shuffle) == bool
        assert random_state is None or type(random_state) == int or isinstance(random_state, np.random.RandomState)
        
        # save the variables
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = np.random.RandomState(random_state) if type(random_state) == int else random_state
        
    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Get the amount of splits to be performed
        :param X:      (obj) Always ignored, exists for compatibility
        :param y:      (obj) Always ignored, exists for compatibility
        :param groups: (obj) Always ignored, exists for compatibility
        :return:       (int) number of splitting iterations in the cross-validator
        """
        return self.n_splits
    
    def split(self, X, y, groups=None):
        """
        Create a generator that return indexes of evenly split classes of multi label y
        :param X:      (obj) Training data, where n_samples is the number of samples and n_features is the number of features
        :param y:      (obj) The target variable for supervised learning problems. Stratification is done based on the y labels
        :param groups: (obj) Always ignored, exists for compatibility
        :yield: The training set indices for that split / The testing set indices for that split
        """
        assert type(y) is np.ndarray
        assert np.issubdtype(y.dtype, np.integer)
        assert len(y.shape) == 2
        assert y.shape[0] > self.n_splits
        assert y.shape[1] > 1
        
        # create train data frame
        df = pd.DataFrame(data=np.array([np.arange(y.shape[0]), np.empty(y.shape[0], dtype=object)]).T, columns=['X', 'Y'])
        for i in range(y.shape[0]):
            df.at[i, 'Y'] = y[i]
        
        # set the target labels
        df['Target'] = df['Y'].apply(lambda x: ' '.join([str(i) for i in x if i == 1]))
        
        # do a value count of the target labels
        vc = df['Target'].value_counts()
        
        # if shuffle
        if self.shuffle:
            for sp in range(self.n_splits):
                # set the train and test lists
                train = list()
                test = list()

                # for the higher represented classes
                for target in vc[vc >= self.n_splits].index:
                    # filter images
                    images = df[df['Target'] == target]

                    # shuffle the array 
                    indexes = images['X'].values
                    if self.random_state is None:
                        np.random.shuffle(indexes) 
                    else:
                        self.random_state.shuffle(indexes)
                        
                    # select the first y_shape*(n_split-1)/n_splits as train and the other y_shape/n_splits as test
                    i = int(np.round(images.shape[0]/self.n_splits))
                    i1 = indexes[:(self.n_splits - 1)*i]
                    i2 = indexes[(self.n_splits - 1)*i:]
                    if len(i1) > 0:
                        train.append(i1)
                    if len(i2) > 0:
                        test.append(i2)

                # filter the low represented classes
                pool = df[df['Target'].isin(vc[vc < self.n_splits].index)]

                # go through each class
                for target in np.argsort(df['Y'].sum()):
                    # filter pool
                    images = pool[pool['Y'].apply(lambda x: x[target] == 1)]
                    if images.shape[0] == 0:
                        continue

                    # shuffle the array 
                    indexes = images['X'].values
                    if self.random_state is None:
                        np.random.shuffle(indexes) 
                    else:
                        self.random_state.shuffle(indexes)
                        
                    # select the first y_shape*(n_split-1)/n_splits as train and the other y_shape/n_splits as test
                    i = int(np.round(images.shape[0]/self.n_splits))
                    i1 = indexes[:(self.n_splits - 1)*i]
                    i2 = indexes[(self.n_splits - 1)*i:]
                    if len(i1) > 0:
                        train.append(i1)
                    if len(i2) > 0:
                        test.append(i2)

                    # remove from pool
                    pool = pool[~pool['X'].isin(images['X'])]

                # stack the train and test indexes
                train = np.concatenate(train)
                test = np.concatenate(test)

                # yield indexes
                yield train, test
        
        # if not shuffle
        else:
            # set the train and test lists
            dataset = [list() for i in range(self.n_splits)]
            
            # for the higher represented classes
            for target in vc[vc > self.n_splits].index:
                # filter images
                images = df[df['Target'] == target]

                # shuffle the array 
                indexes = images['X'].values
                if self.random_state is None:
                    np.random.shuffle(indexes) 
                else:
                    self.random_state.shuffle(indexes)
                
                # calculate the size of indexes to split array
                i = int(np.round(images.shape[0]/self.n_splits))
                
                # go throw each split
                for s in range(1, self.n_splits):
                    # filter indexes by split size
                    ds = indexes[(s - 1)*i:s*i]
                    
                    # If the array is not empty
                    if len(ds) > 0:
                        # add the indexes to the dataset part
                        dataset[s - 1].append(ds)
                
                # for the final split add the rest of the dataset
                ds = indexes[(self.n_splits - 1)*i:]
                if len(ds) > 0:
                    dataset[(self.n_splits - 1)].append(ds)
                
            # filter the low represented classes
            pool = df[df['Target'].isin(vc[vc <= self.n_splits].index)]

            # go through each class
            for target in np.argsort(df['Y'].sum()):
                # filter pool
                images = pool[pool['Y'].apply(lambda x: x[target] == 1)]
                if images.shape[0] == 0:
                    continue

                # shuffle the array 
                indexes = images['X'].values
                if self.random_state is None:
                    np.random.shuffle(indexes) 
                else:
                    self.random_state.shuffle(indexes)
                        
                # calculate the size of indexes to split array
                i = int(np.round(images.shape[0]/self.n_splits))
                
                # go throw each split
                for s in range(1, self.n_splits):
                    # filter indexes by split size
                    ds = indexes[(s - 1)*i:s*i]
                    
                    # If the array is not empty
                    if len(ds) > 0:
                        # add the indexes to the dataset part
                        dataset[s - 1].append(ds)
                
                # for the final split add the rest of the dataset
                ds = indexes[(self.n_splits - 1)*i:]
                if len(ds) > 0:
                    dataset[(self.n_splits - 1)].append(ds)

                # remove from pool
                pool = pool[~pool['X'].isin(images['X'])]
            
            # stack the datasets
            f_dataset = [np.concatenate(d) for d in dataset]
            
            # go throw eac
            for sp in range(self.n_splits):
                # yield indexes
                yield np.concatenate(f_dataset[:sp] + f_dataset[sp + 1:]), f_dataset[sp]

    
class DataGenerator(keras.utils.Sequence):
    """
    Extend the Sequence class from the keras.utils module to create a class
    capable of loading the images from the zipfile in batches, resizing it and
    selecting specific channels

    source: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def __init__(self, train_data, batch_size, file_path, shape, augment_flag=True):
        assert isinstance(train_data, pd.DataFrame)
        assert train_data.shape[0] > 0
        assert batch_size > 0
        assert (os.path.isfile(file_path) and '.zip' in file_path) or os.path.isdir(file_path)
        assert isinstance(shape, Iterable)
        assert len(shape) == 3
        assert type(augment_flag) == bool

        # saved arguments
        self.train_data = train_data
        self.batch_size = batch_size
        self.file_path = file_path
        self.shape = shape
        self.augment_flag = augment_flag

        # get the number of images in the train dataset
        self.size = train_data.shape[0]

        # get list of images
        self.image_ids = train_data['Id'].values

        # set a list of indexes to be extracted from the image
        self.indexes = np.arange(len(self.image_ids))

        # calculate the required number of batches
        self.batches = int(np.floor(self.size / batch_size))

    def __len__(self):
        return self.batches

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: (int) index of the total batches size to be loaded
        :return:
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # select images ids
        batch_ids = [self.image_ids[k] for k in indexes]

        # create array to hold images
        batch_images = np.empty((self.batch_size, self.shape[0], self.shape[1], self.shape[2]))
        batch_labels = np.zeros((self.batch_size, 28))

        # load images into array
        for i in range(self.batch_size):
            # apply transformations to images based on augment flag
            if self.augment_flag:
                batch_images[i] = augment(self.__load_image(batch_ids[i]))
            else:
                batch_images[i] = self.__load_image(batch_ids[i])
            batch_labels[i] = self.train_data.loc[self.train_data['Id'] == batch_ids[i], 'Classes'].values[0]

        # return the images and batches
        return batch_images, batch_labels

    def __iter__(self):
        """
        Create a generator that iterate over the Sequence
        """
        for i in range(self.batches):
            yield self[i]

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.image_ids))
        np.random.shuffle(self.indexes)

    def __load_image(self, image_name):
        """
        Load image contained within a zip file
        :param image_name: (string) image name contained in zip file
        :return:           (array) 3D of RGBY divided by 255
        """
        if '.zip' in self.file_path:
            # open the zipfile
            with ZipFile(self.file_path) as z:
                # load images by channel
                channel_list = list()
                for c in ['red', 'green', 'blue', 'yellow']:
                    with z.open(image_name + '_' + c + '.png') as file:
                        img = cv2.imread(file, 0)
                        img = cv2.resize(img, dsize=(self.shape[0], self.shape[1]), interpolation=cv2.INTER_CUBIC)
                        img = np.array(img)
                        channel_list.append(img)

        else:
            # load images by channel
            channel_list = list()
            for c in ['red', 'green', 'blue', 'yellow']:
                img = cv2.imread(self.file_path + '/' + image_name + '_' + c + '.png', 0)
                img = cv2.resize(img, dsize=(self.shape[0], self.shape[1]), interpolation=cv2.INTER_CUBIC)
                img = np.array(img)
                channel_list.append(img)
        
        # stack pixels of image
        if self.shape[2] == 3:
            image = np.stack((
                channel_list[0]/2 + channel_list[3]/2, 
                channel_list[1]/2 + channel_list[3]/2, 
                channel_list[2]
            ), -1)
        else:
            image = np.stack(channel_list, -1)
        
        # normalize pixels range
        image = np.divide(image, 255)
        
        # return array with normalized colors
        return image


# ## Constants definition

# In[ ]:


# ##### GLOBAL SETTINGS ##### #
# train set file path
TRAIN_PATH = '/kaggle/input/human-protein-atlas-image-classification/train'
# test file path
TEST_PATH = '/kaggle/input/human-protein-atlas-image-classification/test'
# ratio of total train set that is going to be used as test
TEST_SIZE = 0.2
# flag indicating if we should shuffle on K-Fold validation
SHUFFLE = False
# random state
RANDOM_STATE = 42
# model verbose flag
VERBOSE = 1
# check point verbose mode
VERBOSE_CK = 2
# test batch size
TEST_BATCH = 256
# loss function applied
LOSS = focal_loss() # 'binary_crossentropy'
# metrics mesured in model
METRICS = ['accuracy', f1]

# ##### MODEL SPECIFIC ##### #
# shape of image to go into model
INPUT_SHAPE = (299, 299, 3)
# model name
MODEL_NAME = 'InceptionResNet'

# ##### GRID ADJUSTABLE ##### #
# size of train batch
TRAIN_BATCH = 10
# optimizer to be applied to model
OPTIMIZER = Adam(1e-3)
# total number of epochs
EPOCHS = 15
# number of steps per epoch
STEPS_PER_EPOCH = 100
# total number of validation steps
VALIDATION_STEPS = 50


# ## Data processing

# #### Load and adjust train dataset

# In[ ]:


print('LOADING TRAIN CSV FILE')
train_csv = pd.read_csv('/kaggle/input/human-protein-atlas-image-classification/train.csv')

print('OBTAINING CLASSES ARRAY')
train_csv['Classes'] = train_csv['Target'].apply(get_classes)

print('ADJUSTING Y LABELS')
y = MultiLabelBinarizer().fit_transform(train_csv['Target'].str.split(' '))


#  #### Check if split is even accross classes results
d = {
    'Base': train_csv['Classes'].sum(),
    'Train': train_df['Classes'].sum(),
    'Test': test_df['Classes'].sum()
}
df = pd.DataFrame(d)
df['Train'] = df['Train']/df['Base']
df['Test'] = df['Test']/df['Base']
# ## Train model

# In[ ]:


keras.backend.clear_session()

print('DEFINING CNN MODEL')
if MODEL_NAME == 'Sequential':
    model = sequencial_model(INPUT_SHAPE, 28)
elif MODEL_NAME == 'InceptionResNet':
    model = inception_res_net_model(INPUT_SHAPE, 28)
elif MODEL_NAME == 'ResNet50':
    model = res_net_50(INPUT_SHAPE, 28)
elif MODEL_NAME == 'VGG19':
    model = vgg_model(INPUT_SHAPE, 28)


# ### Train baseline model
print('COMPILING MODEL')
model.compile(optimizer=Adam(1e-3), loss=LOSS, metrics=METRICS)
model.summary()

print('CREATING TRAN AND TEST GENERATORS')
kf = StratifiedMultiLabelKFold(int(round(1/TEST_SIZE)), shuffle=SHUFFLE, random_state=RANDOM_STATE)
kf_gen = kf.split(None, y)
train, test = next(kf_gen)
train_gen = DataGenerator(train_csv.iloc[train][['Id', 'Classes']], TRAIN_BATCH, TRAIN_PATH, INPUT_SHAPE)
test_gen = DataGenerator(train_csv.iloc[test][['Id', 'Classes']], TEST_BATCH, TRAIN_PATH, INPUT_SHAPE, augment_flag=False)
class_weights = dict(zip(range(28), max(train_csv['Classes'].sum())/train_csv['Classes'].sum()))

print('CREATING MODEL CHECK POINT')
dt = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
model_path = '/kaggle/working/' + MODEL_NAME + dt + '.model'
checkpoint = ModelCheckpoint(model_path, verbose=VERBOSE_CK, save_best_only=True)

print('FITTING MODEL')
hist = model.fit_generator(
    generator=train_gen, 
    validation_data=test_gen,
    
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    
    class_weight=class_weights,
    
    verbose=VERBOSE,
    callbacks=[checkpoint]
)show_history(hist)
# ### Perform grid search

# In[ ]:


# define parameters
grid = {
    'epochs': [30, 50, 100],
    'steps_per_epoch': [100, 150, 200],
    'learning_rate': [0.001, 0.0005, 0.0001],
    'loss': ['binary_crossentropy', 'focal_loss'],
    'batch_size': [10, 30, 50],
    'class_weight': [None, 0.5, 0.3]
}

dt = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
results_path = '/kaggle/working/' + 'GS_' + MODEL_NAME + dt + '.pkl'

# Do train test split
kf = StratifiedMultiLabelKFold(int(round(1/TEST_SIZE)), shuffle=SHUFFLE, random_state=RANDOM_STATE)
kf_gen = kf.split(None, y)
train, test = next(kf_gen)

# create combination of parameters
params = sorted(grid)
combinations = list(itertools.product(*(grid[key] for key in params)))

# set results data frame
results = list()
r = 0

# for each combination
for comb in combinations:
    # set combination dictionary
    c = dict(zip(params, comb))
    print('RUNNING FOR', c)
    
    print('    CREATING PARAMETERS DICTIONARY')
    # create a dictionary of model parameters
    model_params = {'metrics': METRICS}
    fit_params = {'validation_steps': VALIDATION_STEPS, 'verbose': VERBOSE}
    train_params = {'file_path': TRAIN_PATH, 'shape': INPUT_SHAPE, 'augment_flag': True}
    
    # for each parameter
    for i in range(len(params)):
        # if the parameter is one of the fit ones
        if params[i] in ['epochs', 'steps_per_epoch']:
            # add this parameter to the fit params dictionary
            fit_params[params[i]] = comb[i]
        
        # if the parameter is one of the model ones
        elif params[i] in ['loss']:
            # add this parameter to the model params dictionary
            if comb[i] == 'focal_loss':
                model_params[params[i]] = focal_loss()
            else:
                model_params[params[i]] = comb[i]
        
        # if the parameter is one of the model ones
        elif params[i] in ['learning_rate']:
            # add this parameter to the model params dictionary
            model_params['optimizer'] = Adam(comb[i])
            
        # if the parameter is one of the weighting ones
        elif params[i] in ['class_weight']:
            if comb[i] is None:
                continue
            weight = max(train_csv['Classes'].sum()) / train_csv['Classes'].sum()
            weight = weight / weight.sum()
            csort = np.argsort(weight)
            csum = np.cumsum(weight[csort])
            weight[csort[np.where(csum < 0.5)]] = weight[csort[np.where(csum < 0.5)]] * (1 - comb[i])
            weight[csort[np.where(csum >= 0.5)]] = weight[csort[np.where(csum >= 0.5)]] * comb[i]
            fit_params[params[i]] = 10000 * weight

        # if the parameter is one of the trains parameter
        elif params[i] in ['batch_size']:
            # add this to the train param
            TRAIN_BATCH = comb[i]
    
    print('    SETTING TRAIN AND TEST SETS')
    
    # create train and test generators
    train_gen = DataGenerator(train_csv.iloc[train][['Id', 'Classes']], TRAIN_BATCH, TRAIN_PATH, INPUT_SHAPE)
    test_gen = DataGenerator(train_csv.iloc[test][['Id', 'Classes']], TEST_BATCH, TRAIN_PATH, INPUT_SHAPE, augment_flag=False)
    
    print('    COMPILING MODEL')
    # compile model
    model.compile(**model_params)
    
    print('    CREATING MODEL CHECK POINT')
    dt = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    model_path = '/kaggle/working/' + 'GS_' + MODEL_NAME + dt + '.model'
    checkpoint = ModelCheckpoint(model_path, verbose=VERBOSE_CK, save_best_only=True)
    
    print('    FITTING MODEL')
    # add generators to fit param
    fit_params['generator'] = train_gen
    fit_params['validation_data'] = test_gen
    
    # add checkpoint to fit param
    fit_params['callbacks'] = [checkpoint]
    
    # fit model
    hist = model.fit_generator(**fit_params)
    
    print('    RE-LOAD BEST MODEL')
    model = load_model(
        model_path, 
        custom_objects={'f1': f1, 'focal_loss_fixed': focal_loss()}
    )

    print('    PREDICTING ON TEST SET')
    # create array of predictions
    y_pred = np.array([model.predict(load_image(TRAIN_PATH, name, shape=INPUT_SHAPE)[np.newaxis])[0] for name in train_csv.iloc[test]['Id']])
    
    print('    EVALUATING MODEL')
    results.append(list())

    # add the combination
    results.append(c)

    # add predictions
    results.append(pd.DataFrame(np.concatenate((train_csv.iloc[test]['Id'].values[:, None], y_pred), axis=1)))

    # set the y test value
    y_true = MultiLabelBinarizer().fit_transform(train_csv.iloc[test]['Target'].str.split(' '))

    # apply the precision recall and f1 score
    results.append(pd.DataFrame(np.stack(precision_recall_fscore_support(y_true, (y_pred > 0.2).astype(np.uint8)), -1)))

    # add the accuracy
    results.append(accuracy_score(y_true, (y_pred > 0.2).astype(np.uint8)))

    r += 1

    print('    EXPORTING PARTIAL RESULTS')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)


# ### Train especific model on pre selected parameters

# In[ ]:





# ## Create submit

# ### Load best model

# In[ ]:


model_path = '/kaggle/working/' + MODEL_NAME + dt + '.model'
model = load_model(
    model_path, 
    custom_objects={'f1': f1}
)


# ### Load submit

# In[ ]:


submit = pd.read_csv('/kaggle/input/human-protein-atlas-image-classification/sample_submission.csv')


# ### Predict for test set

# In[ ]:


get_ipython().run_cell_magic('time', '', "predicted = []\nfor name in submit['Id'].values.flatten():\n    image = load_image(TEST_PATH, name, shape=INPUT_SHAPE)\n    score_predict = model.predict(image[np.newaxis])[0]\n    label_predict = np.arange(28)[score_predict >= 0.2]\n    str_predict_label = ' '.join(str(l) for l in label_predict)\n    predicted.append(str_predict_label)")


# In[ ]:


submit['Predicted'] = predicted


# In[ ]:


submit.to_csv('submission.csv', index=False)

