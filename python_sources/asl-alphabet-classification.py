#!/usr/bin/env python
# coding: utf-8

# # Paper of AlaxNet: [ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

# # Importing Library & Global Setting

# In[ ]:


import numpy as np
import pandas as pd
import pickle
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Add, MaxPool2D,                             GlobalAveragePooling2D, DepthwiseConv2D, BatchNormalization, LeakyReLU
from keras import regularizers
from keras.models import Sequential, load_model, Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import print_summary, plot_model, to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta, RMSprop
from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.applications.inception_v3 import InceptionV3
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from numpy.random import seed
from tensorflow import set_random_seed

import cv2
from glob import glob
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random
from math import ceil

import seaborn as sns
sns.set()
import itertools

import time, datetime, pytz

import sys
import os
from os import getenv
from pathlib import Path


def print_time(end='\n'):
    dt = datetime.datetime.fromtimestamp(time.time()).astimezone(pytz.timezone('Asia/Hong_Kong'))
    print("[%s] " % dt, end=end)
    
def get_datetime():
    dt = datetime.datetime.fromtimestamp(time.time()).astimezone(pytz.timezone('Asia/Hong_Kong'))
    return dt.strftime("%Y-%m-%d_%H.%M.%S")

SEED = 1
seed(SEED)
set_random_seed(SEED)

SAVE_OUTPUT_TO_FILE = True
SAVE_MODEL_TO_FILE = True
RELOAD_MODEL_FROM_FILE = False

exec_time = get_datetime()

MODEL_NAME = exec_time + '-model'
MODEL_DIR = '.'
MODEL_SAVE_DIR = MODEL_DIR + '/' + MODEL_NAME + '.h5'
MODEL_SAVE_WEIGHTS_DIR = MODEL_DIR + '/' + MODEL_NAME + '.weights.h5'
MODEL_SAVE_TRAIN_LOG_DIR = MODEL_DIR + '/' + MODEL_NAME + '-train-log.pickle'

RELOAD_MODEL_NAME = '2019-04-09_12.15.56-model'
RELOAD_MODEL_BASE = '../input/alaxnet-pretrained-model'
RELOAD_MODEL_DIR = RELOAD_MODEL_BASE + '/' + RELOAD_MODEL_NAME + '.h5'
RELOAD_MODEL_WEIGHTS_DIR = RELOAD_MODEL_BASE + '/' + RELOAD_MODEL_NAME + '.weights.h5'
RELOAD_MODEL_TRAIN_LOG_DIR = RELOAD_MODEL_BASE + '/' + RELOAD_MODEL_NAME + '-train-log.pickle'

DEFAULT_STDOUT = sys.stdout
TRAIN_DIR = '../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train'
TEST_DIR = '../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test'
CUSTOM_TEST_DIR = '../input/asl-alphabet-test/asl-alphabet-test'
CLASSES = [os.path.basename(folder) for folder in glob(TRAIN_DIR + '/*')]
CLASSES.remove('J')
CLASSES.remove('Z')
CLASSES.sort()

BATCH_SIZE, HEIGHT, WIDTH, CHANNELS = 64, 64, 64, 3
SAMPLE_SIZE = (HEIGHT, WIDTH)
INPUT_DIMS = (HEIGHT, WIDTH, CHANNELS)
NUM_CLASSES = len(CLASSES)
VALIDATION_SPLIT = 0.1

print("Classes: %s \n" % CLASSES)

print("Use the reloaded model: %s" % ("Yes" if RELOAD_MODEL_FROM_FILE else "No"))
if RELOAD_MODEL_FROM_FILE:
    print("Reloading from \n  * \"%s\" \n  * \"%s\" \n" 
          % (RELOAD_MODEL_DIR, RELOAD_MODEL_WEIGHTS_DIR))


print("Save Output to File: %s" % ("Yes" if SAVE_OUTPUT_TO_FILE else "No"))
if SAVE_OUTPUT_TO_FILE:
    OUT_FILE_NAME = exec_time+'_out.txt'
    print("Save Output to %s \n" % OUT_FILE_NAME)
#     sys.stdout = open(OUT_FILE_NAME, 'w+')


# # Sample Training Data

# In[ ]:


def show_one_random_image_per_class(DIR):
    cols = 6
    rows = int(ceil(NUM_CLASSES/cols))
    fig = plt.figure(figsize=(12, 11))
    fig.suptitle("Sample Images from \"%s\"" % DIR, fontsize=16)
    for i in range(NUM_CLASSES):
        image_folder = DIR + "/" + CLASSES[i] + "/**"
        images_path = glob(image_folder)
        random_img = random.choice(images_path)
        sp = plt.subplot(rows, cols, i + 1)
        plt.imshow(mpimg.imread(random_img))
        plt.title(CLASSES[i])
        sp.axis('off')

show_one_random_image_per_class(TRAIN_DIR)


# # Preprocessing Data

# In[ ]:


# labels_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,
#                'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,
#                'Z':25,'space':26,'del':27,'nothing':28}
labels_dict = {}
i = 0
for char in CLASSES:
    labels_dict[char] = i
    i += 1

def load_data_old(DIR, VALI_SPLIT):
    train_dir = DIR
    images = []
    labels = []
    size = HEIGHT,WIDTH
    print("LOADING DATA FROM : ",end = "")
    for folder in sorted(os.listdir(train_dir)):
        if folder in ['J', 'Z']:
            continue
        print(folder, end = ' | ')
        for image in os.listdir(train_dir + "/" + folder):
            temp_img = cv2.imread(train_dir + '/' + folder + '/' + image)
            temp_img = cv2.resize(temp_img, size)
            images.append(temp_img)
            if folder == 'A':
                labels.append(labels_dict['A'])
            elif folder == 'B':
                labels.append(labels_dict['B'])
            elif folder == 'C':
                labels.append(labels_dict['C'])
            elif folder == 'D':
                labels.append(labels_dict['D'])
            elif folder == 'E':
                labels.append(labels_dict['E'])
            elif folder == 'F':
                labels.append(labels_dict['F'])
            elif folder == 'G':
                labels.append(labels_dict['G'])
            elif folder == 'H':
                labels.append(labels_dict['H'])
            elif folder == 'I':
                labels.append(labels_dict['I'])
#             elif folder == 'J':
#                 labels.append(labels_dict['J'])
            elif folder == 'K':
                labels.append(labels_dict['K'])
            elif folder == 'L':
                labels.append(labels_dict['L'])
            elif folder == 'M':
                labels.append(labels_dict['M'])
            elif folder == 'N':
                labels.append(labels_dict['N'])
            elif folder == 'O':
                labels.append(labels_dict['O'])
            elif folder == 'P':
                labels.append(labels_dict['P'])
            elif folder == 'Q':
                labels.append(labels_dict['Q'])
            elif folder == 'R':
                labels.append(labels_dict['R'])
            elif folder == 'S':
                labels.append(labels_dict['S'])
            elif folder == 'T':
                labels.append(labels_dict['T'])
            elif folder == 'U':
                labels.append(labels_dict['U'])
            elif folder == 'V':
                labels.append(labels_dict['V'])
            elif folder == 'W':
                labels.append(labels_dict['W'])
            elif folder == 'X':
                labels.append(labels_dict['X'])
            elif folder == 'Y':
                labels.append(labels_dict['Y'])
#             elif folder == 'Z':
#                 labels.append(labels_dict['Z'])
            elif folder == 'space':
                labels.append(labels_dict['space'])
            elif folder == 'del':
                labels.append(labels_dict['del'])
            elif folder == 'nothing':
                labels.append(labels_dict['nothing'])
    
    images = np.array(images)
#     images = images.astype('float32')
    images = images.astype('float32')/255.0
#     images = preprocess_input(images)
    
    labels = to_categorical(labels)
    
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size = VALI_SPLIT)
    
    print()
    print('Loaded', len(X_train),'images for training,','Train data shape =',X_train.shape)
    print('Loaded', len(X_test),'images for testing','Test data shape =',X_test.shape)
    
    return X_train, X_test, Y_train, Y_test


# In[ ]:


# labels_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,
#                'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,
#                'Z':25,'space':26,'del':27,'nothing':28}

# def load_data_new(DIR, VALI_SPLIT):
#     data_generator = ImageDataGenerator(rotation_range=15,
#                                     width_shift_range=0.1,
#                                     height_shift_range=0.1,
#                                     shear_range=0.1,
#                                     zoom_range=0.1,
#                                     horizontal_flip=True,
#                                     fill_mode='constant')
#     train_dir = DIR
#     images = []
#     labels = []
#     size = HEIGHT,WIDTH
#     print("LOADING DATA FROM : ",end = "")
#     for folder in sorted(os.listdir(train_dir)):
# #         if folder == 'C':
# #             break
#         print(folder, end = ' | ')
#         for image in os.listdir(train_dir + "/" + folder):
#             temp_img = cv2.imread(train_dir + '/' + folder + '/' + image)
#             temp_img = cv2.resize(temp_img, size)
#             temp_img = np.array(temp_img)
#             temp_img = temp_img.astype('float32')/255.0
#             images.append(temp_img.tolist())
#             if folder == 'A':
#                 labels.append(labels_dict['A'])
#             elif folder == 'B':
#                 labels.append(labels_dict['B'])
#             elif folder == 'C':
#                 labels.append(labels_dict['C'])
#             elif folder == 'D':
#                 labels.append(labels_dict['D'])
#             elif folder == 'E':
#                 labels.append(labels_dict['E'])
#             elif folder == 'F':
#                 labels.append(labels_dict['F'])
#             elif folder == 'G':
#                 labels.append(labels_dict['G'])
#             elif folder == 'H':
#                 labels.append(labels_dict['H'])
#             elif folder == 'I':
#                 labels.append(labels_dict['I'])
#             elif folder == 'J':
#                 labels.append(labels_dict['J'])
#             elif folder == 'K':
#                 labels.append(labels_dict['K'])
#             elif folder == 'L':
#                 labels.append(labels_dict['L'])
#             elif folder == 'M':
#                 labels.append(labels_dict['M'])
#             elif folder == 'N':
#                 labels.append(labels_dict['N'])
#             elif folder == 'O':
#                 labels.append(labels_dict['O'])
#             elif folder == 'P':
#                 labels.append(labels_dict['P'])
#             elif folder == 'Q':
#                 labels.append(labels_dict['Q'])
#             elif folder == 'R':
#                 labels.append(labels_dict['R'])
#             elif folder == 'S':
#                 labels.append(labels_dict['S'])
#             elif folder == 'T':
#                 labels.append(labels_dict['T'])
#             elif folder == 'U':
#                 labels.append(labels_dict['U'])
#             elif folder == 'V':
#                 labels.append(labels_dict['V'])
#             elif folder == 'W':
#                 labels.append(labels_dict['W'])
#             elif folder == 'X':
#                 labels.append(labels_dict['X'])
#             elif folder == 'Y':
#                 labels.append(labels_dict['Y'])
#             elif folder == 'Z':
#                 labels.append(labels_dict['Z'])
#             elif folder == 'space':
#                 labels.append(labels_dict['space'])
#             elif folder == 'del':
#                 labels.append(labels_dict['del'])
#             elif folder == 'nothing':
#                 labels.append(labels_dict['nothing'])
    
# #     num_gen_img = 1
# #     X = img_to_array(temp_img)
# #     X = X.reshape((1,) + X.shape)
# #     # Apply transformation
# #     i = 0
# #     for batch in data_generator.flow(X):
# #         i += 1
# # #         plt.imshow(array_to_img(batch[0]))
# # #         plt.show()
# #         images.append(cv2.resize(batch[0], size))
# #         if i % num_gen_img == 0:  # Generate three transformed pictures
# #             break  # To avoid generator to

#     num_gen_img = 10
#     # Apply transformation
#     i = 0
#     for x_batch, y_batch in data_generator.flow(np.array(images), labels, shuffle=True, batch_size=BATCH_SIZE):
#         i += 1
#         for l in range(len(x_batch)):
# #             plt.imshow(array_to_img(x_batch[l]))
# #             plt.show()
#             images.append(x_batch[l])
#             labels.append(y_batch[l])
# #             print(y_batch[l])
#         if i % num_gen_img == 0:  # Generate three transformed pictures
#             break  # To avoid generator to
            
    
#     images = np.array(images)
# #     images = images.astype('float32')
# #     images = images.astype('float32')/255.0
# #     images = preprocess_input(images)
    
#     labels = to_categorical(labels)
    
#     X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size = VALI_SPLIT)
    
#     print()
#     print('Loaded', len(X_train),'images for training,','Train data shape =',X_train.shape)
#     print('Loaded', len(X_test),'images for testing','Test data shape =',X_test.shape)
    
#     return X_train, X_test, Y_train, Y_test


# In[ ]:


X_train, X_test, Y_train, Y_test = load_data_old(TRAIN_DIR, VALIDATION_SPLIT)
X1_train, X1_test, Y1_train, Y1_test = load_data_old(CUSTOM_TEST_DIR, 0)


# In[ ]:


train_image_generator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     zca_whitening=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True
)
# train_image_generator.fit(X_train)

val_image_generator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     zca_whitening=True,
)
# val_image_generator.fit(X_train)

train_generator = train_image_generator.flow(x=X_train, y=Y_train, batch_size=BATCH_SIZE, shuffle=True)
val_generator = val_image_generator.flow(x=X_test, y=Y_test, batch_size=BATCH_SIZE, shuffle=False)

NUM_TRAIN_SAMPLE = train_generator.n
NUM_VAL_SAMPLE = val_generator.n


# In[ ]:


# def preprocess_image(image):
#     return image # No preprocessing for now

# def image_generator(options):
#     flow_options = {
#         'target_size': options.get('target_size'),
#         'batch_size': options.get('batch_size'),
#         'subset': options.get('subset'),
#         'shuffle': options.get('shuffle'),
#         'seed': SEED,
#     }
    
#     data_dir = options.get('data_dir', TRAIN_DIR)
# #     gen = ImageDataGenerator(data_format='channels_last', dtype=(BATCH_SIZE, HEIGHT, WIDTH, CHANNELS))
# #     data_xy = gen.flow_from_directory(data_dir, **flow_options)
# #     data_xy = data_xy[0].astype('float32')
    
#     gen_options = {
#         'samplewise_center': options.get('samplewise_center', False),
#         'samplewise_std_normalization': options.get('samplewise_std_normalization', False),
#         'featurewise_center': options.get('featurewise_center', False),
#         'featurewise_std_normalization': options.get('featurewise_std_normalization', False),
#         'validation_split': options.get('validation_split'),
#         'preprocessing_function': options.get('preprocessing_function'),
        
# #         'rotation_range': 40,
# #         'width_shift_range': 0.2,
# #         'height_shift_range': 0.2,
# #         'shear_range': 0.2,
# #         'zoom_range': 0.2,
# #         'fill_mode': 'nearest',
        
# #         'data_format': 'channels_last',
# #         'dtype': (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS),
#     }
#     data_augmentor = ImageDataGenerator(**gen_options)
# #     data_augmentor.fit(data_xy[0])
    
#     return data_augmentor.flow_from_directory(data_dir, **flow_options)

# def image_generator_for_training(subset):
#     options = {
#         'samplewise_center': True,
#         'samplewise_std_normalization': True,
# #         'featurewise_center': True,
# #         'featurewise_std_normalization': True,
#         'validation_split': VALIDATION_SPLIT,
#         'preprocessing_function': preprocess_image,
#         'data_dir': TRAIN_DIR,
#         'target_size': SAMPLE_SIZE,
#         'batch_size': BATCH_SIZE,
#         'subset': subset,
#         'shuffle': True,
#     }
#     return image_generator(options)

# def image_generator_for_validating(subset):
#     options = {
#         'samplewise_center': True,
#         'samplewise_std_normalization': True,
# #         'featurewise_center': True,
# #         'featurewise_std_normalization': True,
#         'validation_split': VALIDATION_SPLIT,
#         'preprocessing_function': preprocess_image,
#         'data_dir': TRAIN_DIR,
#         'target_size': SAMPLE_SIZE,
#         'batch_size': BATCH_SIZE,
#         'subset': subset,
#         'shuffle': False,
#     }
#     return image_generator(options)

# def image_generator_for_testing():
#     options = {
#         'samplewise_center': True,
#         'samplewise_std_normalization': True,
# #         'featurewise_center': True,
# #         'featurewise_std_normalization': True,
#         'validation_split': 0.0,
#         'preprocessing_function': preprocess_image,
#         'data_dir': CUSTOM_TEST_DIR,
#         'target_size': SAMPLE_SIZE,
#         'batch_size': BATCH_SIZE,
#         'shuffle': False,
#     }
#     return image_generator(options)

# CUSTOM_TEST_DIR

# train_generator = image_generator_for_training("training")
# val_generator = image_generator_for_training("validation")

# NUM_TRAIN_SAMPLE = train_generator.n
# NUM_VAL_SAMPLE = val_generator.n


# # Reload Learning Model From Disk
# (if RELOAD_MODEL_FROM_FILE == True)

# In[ ]:


def reload_model():
    old_model_file = Path(RELOAD_MODEL_DIR)
    old_weight_file = Path(RELOAD_MODEL_WEIGHTS_DIR)
    old_train_log_file = Path(RELOAD_MODEL_TRAIN_LOG_DIR)
    if old_model_file.is_file() and old_weight_file.is_file() and old_train_log_file.is_file():
        print("Reloading old model, weights and training log from disk")
        model = load_model(RELOAD_MODEL_DIR)
        model.load_weights(RELOAD_MODEL_WEIGHTS_DIR)
        with open(RELOAD_MODEL_TRAIN_LOG_DIR, 'rb') as file:
            train_log = pickle.load(file)
        print("Done!")
        return model, train_log
    else:
        print("Cannot reload the old model, weight and training log from\n  * \"%s\"\n  * \"%s\"\n  * \"%s\"" 
              % (RELOAD_MODEL_DIR, RELOAD_MODEL_WEIGHTS_DIR, RELOAD_MODEL_TRAIN_LOG_DIR))
        print("Please check if the path is correct or not")
        return None, None

if RELOAD_MODEL_FROM_FILE:
    model, train_log = reload_model()


# # Define The Learning Model

# In[ ]:


def define_model():
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=INPUT_DIMS))
    model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))
    model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu'))
    model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


#     model = Sequential()
    
#     model.add(Conv2D(16, kernel_size = [3,3], padding = 'same', activation = 'relu', input_shape = INPUT_DIMS))
#     model.add(Conv2D(32, kernel_size = [3,3], padding = 'same', activation = 'relu'))
#     model.add(MaxPool2D(pool_size = [3,3]))
    
#     model.add(Conv2D(32, kernel_size = [3,3], padding = 'same', activation = 'relu'))
#     model.add(Conv2D(64, kernel_size = [3,3], padding = 'same', activation = 'relu'))
#     model.add(MaxPool2D(pool_size = [3,3]))
    
#     model.add(Conv2D(128, kernel_size = [3,3], padding = 'same', activation = 'relu'))
#     model.add(Conv2D(256, kernel_size = [3,3], padding = 'same', activation = 'relu'))
#     model.add(MaxPool2D(pool_size = [3,3]))
    
#     model.add(BatchNormalization())
    
#     model.add(Flatten())
#     model.add(Dropout(0.5))
#     model.add(Dense(512, activation = 'relu', kernel_regularizer = regularizers.l2(0.001)))
#     model.add(Dense(29, activation = 'softmax'))
    
#     model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ["accuracy"])
# #     model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ["accuracy"])


#     inputs = Input(shape=INPUT_DIMS)
#     net = Conv2D(32, kernel_size=3, strides=1, padding="same")(inputs)
#     net = LeakyReLU()(net)
#     net = Conv2D(32, kernel_size=3, strides=1, padding="same")(net)
#     net = LeakyReLU()(net)
#     net = Conv2D(32, kernel_size=3, strides=2, padding="same")(net)
#     net = LeakyReLU()(net)

#     net = Conv2D(32, kernel_size=3, strides=1, padding="same")(net)
#     net = LeakyReLU()(net)
#     net = Conv2D(32, kernel_size=3, strides=1, padding="same")(net)
#     net = LeakyReLU()(net)
#     net = Conv2D(32, kernel_size=3, strides=2, padding="same")(net)
#     net = LeakyReLU()(net)

#     shortcut = net

#     net = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')(net)
#     net = BatchNormalization(axis=3)(net)
#     net = LeakyReLU()(net)
#     net = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(net)
#     net = BatchNormalization(axis=3)(net)
#     net = LeakyReLU()(net)

#     net = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')(net)
#     net = BatchNormalization(axis=3)(net)
#     net = LeakyReLU()(net)
#     net = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(net)
#     net = BatchNormalization(axis=3)(net)
#     net = LeakyReLU()(net)

#     net = Add()([net, shortcut])

#     net = GlobalAveragePooling2D()(net)
#     net = Dropout(0.2)(net)

#     net = Dense(128, activation='relu')(net)
#     outputs = Dense(NUM_CLASSES, activation='softmax')(net)

#     model = Model(inputs=inputs, outputs=outputs)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])


#     map_characters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
#                       10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
#                       19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 
#                       26: 'del', 27: 'nothing', 28: 'space', 29: 'other'}
#     map_characters1 = map_characters
#     Y_train_1 = np.argmax(Y_train, axis=1)
#     class_weight1 = class_weight.compute_class_weight('balanced', np.unique(Y_train_1), Y_train_1)
#     weight_path1 = '../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
#     weight_path2 = '../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
#     pretrained_model_1 = VGG16(weights = weight_path1, include_top=False, input_shape=(HEIGHT, WIDTH, 3))
#     # pretrained_model_2 = InceptionV3(weights = weight_path2, include_top=False, input_shape=(HEIGHT, WIDTH, 3))
#     optimizer1 = Adam()
#     optimizer2 = RMSprop(lr=0.0001)

#     optimizer = optimizer2
#     base_model = pretrained_model_1 # Topless
    
#     # Add top layer
#     x = base_model.output
#     x = Flatten()(x)
#     predictions = Dense(NUM_CLASSES, activation='softmax')(x)
#     model = Model(inputs=base_model.input, outputs=predictions)
    
#     # Train top layer
#     for layer in base_model.layers:
#         layer.trainable = False
#     model.compile(loss='categorical_crossentropy', 
#                   optimizer=optimizer, 
#                   metrics=['accuracy'])
    
    if SAVE_MODEL_TO_FILE:
        model.save(MODEL_SAVE_DIR)
    return model

if not RELOAD_MODEL_FROM_FILE:
    print("Model definded as followings")
    model = define_model()
else:
    print("Use the reloaded model")

print_summary(model)
plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)


# # Training

# In[ ]:


def train(model, train_generator, val_generator, save_to_file):
    callbacks = [
#         TensorBoard(log_dir='./logs/%s' % (start_time)),
#         ModelCheckpoint(MODEL_SAVE_WEIGHTS_DIR, monitor='val_loss', verbose=1, 
#                         save_best_only=True, save_weights_only=True, mode='auto'),
#         EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, 
#                       mode='auto', baseline=None, restore_best_weights=False),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, 
                          patience=5, verbose=1, mode='auto'), # min_lr=0.001
    ]
    train_log = model.fit_generator(train_generator, 
                                    epochs=30,
                                    steps_per_epoch=int(ceil(NUM_TRAIN_SAMPLE/BATCH_SIZE)),
                                    validation_data=val_generator,
                                    validation_steps=int(ceil(NUM_VAL_SAMPLE/BATCH_SIZE)),
                                    verbose=1, # 0=silent, 1=progress bar, 2=one line per epoch
                                    use_multiprocessing=False,
                                    workers=0,
                                    callbacks=callbacks)
    if SAVE_MODEL_TO_FILE: 
        model.save_weights(MODEL_SAVE_WEIGHTS_DIR)
        with open(MODEL_SAVE_TRAIN_LOG_DIR, 'wb') as file:
            pickle.dump(train_log, file)
    return train_log

if not RELOAD_MODEL_FROM_FILE:
    print("Start training the definded model \n")
    start_time = time.time()
    time_limit_sec = 1 * 60**2
    est_running_time_scale = 2
    i = 1
    train_log = None
    if SAVE_OUTPUT_TO_FILE:
        sys.stdout = open(OUT_FILE_NAME, 'a+')
    while(True):
        print_time()
        print("Step %d" % i)
        tic = time.time()
        train_log = train(model, train_generator, val_generator, SAVE_MODEL_TO_FILE)
        toc = time.time()
        running_time = toc - tic
        est_time = running_time * est_running_time_scale
        time_elapsed = toc - start_time
        remaining_time = time_limit_sec - time_elapsed
        print_time()
        print("Step %d, takes %ds; Next Ite Est Time %ds; Time Elapsed %ds; Remaining Time %ds; \n\n" % 
              (i, running_time, est_time, time_elapsed, remaining_time))
        if ( remaining_time < est_time ):
            break
        i += 1
        break
    if SAVE_OUTPUT_TO_FILE:
        sys.stdout = DEFAULT_STDOUT
else:
    print("No training is needed for the reloaded model")


# # Evaluate The Result

# In[ ]:


def plot_confusion_matrix(cm, classes,
                      normalize=False,
                      title='Confusion matrix',
                      cmap=plt.cm.Blues):
    '''
    Plot a confusion matrix heatmap using matplotlib. This code was obtained from
    the scikit-learn documentation:

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return

def plot_confusion_matrix_with_default_options(y_pred, y_true, classes, options=None):
    '''Plot a confusion matrix heatmap with a default size and default options.'''
    cm = confusion_matrix(y_true, y_pred)
    with sns.axes_style('ticks'):
        plt.figure(figsize=(16, 16))
        if options is not None:
            plot_confusion_matrix(cm, classes, **options)
        else:
            plot_confusion_matrix(cm, classes)
        plt.show()
    return


# In[ ]:


def show_incorrect_classified_img(imgs, y_true, y_pred):
    correct_label = y_true==y_pred
    cols = 6
    rows = 1
    num = 1
    fig = plt.figure(figsize=(12, 11))
    for i in range(len(correct_label)):
        if correct_label[i]:
            continue
    #     if i > 10:
    #         break
        sp = plt.subplot(rows, cols, num)
        plt.imshow(imgs[i])
        plt.title("\"%s\" classify as \"%s\"" % (CLASSES[y_true[i]], CLASSES[y_pred[i]]))
        sp.axis('off')
        if num % cols == 0:
            fig = plt.figure(figsize=(12, 11))
            num = 1
        else:
            num += 1


# In[ ]:


# eval_val_generator = image_generator_for_validating("validation")
eval_val_generator = val_image_generator.flow(x=X_test, y=Y_test, batch_size=BATCH_SIZE, shuffle=False)
print()

print("The final classification result on validation test...")
print("loss: %.2f" % train_log.history['val_loss'][-1])
print("acc:  %.2f" % train_log.history['val_acc'][-1])

loss = train_log.history['loss']
val_loss = train_log.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='red', label='Training loss')
plt.plot(epochs, val_loss, color='green', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = train_log.history['acc']
val_acc = train_log.history['val_acc']
plt.plot(epochs, acc, color='red', label='Training acc')
plt.plot(epochs, val_acc, color='green', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


Y_pred = model.predict_generator(eval_val_generator, int(ceil(NUM_VAL_SAMPLE/BATCH_SIZE)))
y_pred = np.argmax(Y_pred, axis=1)
Y_test_1 = np.argmax(Y_test, axis=1)
print('Classification Report')
print(classification_report(y_pred=y_pred, y_true=Y_test_1, target_names=CLASSES))


# In[ ]:


# show_incorrect_classified_img(X_test, Y_test_1, y_pred)


# In[ ]:


with sns.axes_style('ticks'):
    plot_confusion_matrix_with_default_options(y_pred=y_pred, y_true=Y_test_1, classes=CLASSES)
#     plot_confusion_matrix_with_default_options(y_pred=y_pred, y_true=eval_val_generator.classes, classes=CLASSES, 
#                                                options={'normalize': True})


# In[ ]:


# eval_test_generator = image_generator_for_testing()
eval_test_generator = val_image_generator.flow(x=X1_train, y=Y1_train, batch_size=BATCH_SIZE, shuffle=False)

NUM_TEST_SAMPLE = eval_test_generator.n
Y1_pred = model.predict_generator(eval_test_generator, int(ceil(NUM_TEST_SAMPLE/BATCH_SIZE)))
y1_pred = np.argmax(Y1_pred, axis=1)
Y1_test_1 = np.argmax(Y1_train, axis=1)
print('Classification Report')
print(classification_report(y_pred=y1_pred, y_true=Y1_test_1, target_names=CLASSES))


# In[ ]:


# show_incorrect_classified_img(X1_train, Y1_test_1, y1_pred)


# In[ ]:


with sns.axes_style('ticks'):
    plot_confusion_matrix_with_default_options(y_pred=y1_pred, y_true=Y1_test_1, classes=CLASSES)
#     plot_confusion_matrix_with_default_options(y_pred=y_pred, y_true=eval_test_generator.classes, classes=CLASSES, 
#                                                options={'normalize': True})


# In[ ]:


import keras
from keras import models

layers = model.layers[1:-4]
img = X_test[1:2,:,:,:]

# Extracts the outputs of the top 8 layers:
layer_outputs = [layer.output for layer in layers]
# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# This will return a list of 5 Numpy arrays:
# one array per layer activation
activations = activation_model.predict(img)

fig = plt.figure(figsize=(12, 11))
sp = plt.subplot(5, 5, 1)
plt.imshow(img[0])
sp.axis('off')

# These are the names of the layers, so can have them as part of our plot
layer_names = []
for layer in layers:
    layer_names.append(layer.name)

images_per_row = 16

# Now let's display our feature maps
for layer_name, layer_activation in zip(layer_names, activations):
    # This is the number of features in the feature map
    n_features = layer_activation.shape[-1]

    # The feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]

    # We will tile the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # We'll tile each filter into this big horizontal grid
#     print(layer_activation.shape)
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

    # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
plt.show()


# In[ ]:





# In[ ]:





# # References:
# 
# * ImageNet Classification with Deep Convolutional Neural Networks, http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
# * Running Kaggle Kernels with a GPU, https://www.kaggle.com/dansbecker/running-kaggle-kernels-with-a-gpu
# * Classifying Images of the ASL Alphabet using Keras, https://www.kaggle.com/danrasband/classifying-images-of-the-asl-alphabet-using-keras
# * ASL Alphabet Classification with slimCNN, https://www.kaggle.com/kairess/99-9-asl-alphabet-classification-with-slimcnn
# * ASL Classifier using Keras, https://www.kaggle.com/modojj/asl-classifier-using-keras
# * ASL alphabet classification with CNN (Keras), https://www.kaggle.com/dsilvadeepal/asl-alphabet-classification-with-cnn-keras
# * Interpret Sign Language with Deep Learning, https://www.kaggle.com/paultimothymooney/interpret-sign-language-with-deep-learning
# * Visualize CNN with Keras, https://www.kaggle.com/amarjeet007/visualize-cnn-with-keras
# 
