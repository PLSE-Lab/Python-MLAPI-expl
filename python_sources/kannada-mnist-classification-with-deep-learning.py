#!/usr/bin/env python
# coding: utf-8

# ![]()

# # Kannada MNIST

# ![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbEAAAB0CAMAAAA8XPwwAAAC0FBMVEX///8AAAD8/PxMTEycnJz///35///Y2Nj///z///n8//////P///ft//////X3/////+ni4uP///D//+PS0tL//+3a///h///y////+e///9j//+fP//+kpKWk7fS6eDlUoMz5wYT4tGwWWIfc1cPBwcEAADnXxoH328Hq2pH65dF9PDQkAAAAACz//9bh7v1oKjrS5fovAAAAADIZAADJ8v8AAFgAABz/8dv/3bQANlhUMwAAACYAABY8PDy1fVCsazydbz0fO3vD///Gm3GwjFUAAD5NAADo1q7x5r710rFTFj50tMu95f++7/8AKYH8+bM4AADAnITXvZZFjNO12f8AAEZcXFwASIlMQUa05v/rxnurw91BAADO1d6mpbYmACUwcbo0ADCQzP+8n2TDdylPhatdlrfuuoYoJGVHBEEAR5jvvYx9fX18enfGrp0ANnMALFoAQG6fVyjTnW9BMmZzWWVeYnt8s9uNbVWieYx/Y6ORhLW3eH6uscyObpWKVVJxPwRtg7JuN2N6k7N9RVWLWTxcMyg4GwBLaJX94KY3GB1ELhmshEMAX56ZXlaT0tUAG1KgZRN+maioinyur4KDufVQWpFddoapyNPc3sre36lQHGCHQABVkqfhnmSM0/9pTVqWtX6nmcJMY32Jp86Fg6LRnZyPd3d+nLWEHgB0n9myzv9uJACltNdkcavZpJCxblOKcGtzgsOxyMidj393g4AsTmNtk31MMEr/vZotJ0dvUiNRYlTEomS4bC6xxKfD8cR/U3zJqsl/a1GlpG+6gSlLNwDXlpMAFmduW06if1uBg3JpVGyor+dKVnwwZX4dGxB/u79pIixIWWN1SyYbQDlEmeRNADk1ISqIPB9lAABKaGAPXq52TwA+MRx8druieZpNQHU7MAKHrKquXgCTjUSEQ2MZMDSGhV1oXzxIABlQKVMyAFBdRjBeNTnfIpR7AAAgAElEQVR4nO1dj19TV5Z/L/BMAgESAsIDIgJBYlAQFU0E+dWgJEXFBAUxSvEHQpsYEargTK0w1kocKSJTRZaCU0Es9Ue7YtdxandnW2c0Wp2q066Odrdbd/vT/gt7730vv98vZ+x0dyffz6ckNefde+4595x77n333oNhvtBfq8N+TOh3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)

# # 1. Import 

# In[ ]:


# System
import sys
import os
import argparse
import itertools

# Time
import time
import datetime

# Numerical Data
import random
import numpy as np 
import pandas as pd

# Tools
import shutil
from glob import glob
from tqdm import tqdm
import gc

# NLP
import re

# Preprocessing
from sklearn import preprocessing
from sklearn.utils import class_weight as cw
from sklearn.utils import shuffle

# Model Selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Machine Learning Models
from sklearn import svm
from sklearn.svm import LinearSVC, SVC

# Evaluation Metrics
from sklearn import metrics 
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, roc_auc_score


# Deep Learning - Keras -  Preprocessing
from keras.preprocessing.image import ImageDataGenerator

# Deep Learning - Keras - Model
import keras
from keras import models
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential

# Deep Learning - Keras - Layers
from keras.layers import Convolution1D, concatenate, SpatialDropout1D, GlobalMaxPool1D, GlobalAvgPool1D, Embedding,     Conv2D, SeparableConv1D, Add, BatchNormalization, Activation, GlobalAveragePooling2D, LeakyReLU, Flatten
from keras.layers import Dense, Input, Dropout, MaxPool2D, MaxPooling2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D,     Lambda, Multiply, LSTM, Bidirectional, PReLU, MaxPooling1D
from keras.layers.pooling import _GlobalPooling1D

from keras.regularizers import l2

# Deep Learning - Keras - Pretrained Models
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet201
from keras.applications.nasnet import NASNetMobile, NASNetLarge

from keras.applications.nasnet import preprocess_input

# Deep Learning - Keras - Model Parameters and Evaluation Metrics
from keras import optimizers
from keras.optimizers import Adam, SGD , RMSprop
from keras.losses import mae, sparse_categorical_crossentropy, binary_crossentropy

# Deep Learning - Keras - Visualisation
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
# from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K

# Deep Learning - TensorFlow
import tensorflow as tf

# Graph/ Visualization
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.image as mpimg
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix

# Image
import cv2
from PIL import Image
from IPython.display import display

# np.random.seed(42)

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data
print(os.listdir("../input/"))


# # 2. Functions

# In[ ]:


def date_time(x):
    if x==1:
        return 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==2:    
        return 'Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==3:  
        return 'Date now: %s' % datetime.datetime.now()
    if x==4:  
        return 'Date today: %s' % datetime.date.today()  


# # 3. Input Configuration

# In[ ]:


input_directory = r"../input/Kannada-MNIST/"
output_directory = r"../output/"

training_dir = input_directory + "train_images"
testing_dir = input_directory + r"test_images"

if not os.path.exists(output_directory):
    os.mkdir(output_directory)
    
figure_directory = "../output/figures"
if not os.path.exists(figure_directory):
    os.mkdir(figure_directory)

# model_input_directory = "../input/models/"
# if not os.path.exists(model_input_directory):
#     os.mkdir(model_input_directory)

model_output_directory = "../output/models/"
if not os.path.exists(model_output_directory):
    os.mkdir(model_output_directory)


    
file_name_pred_batch = figure_directory+r"/result"
file_name_pred_sample = figure_directory+r"/sample"


# In[ ]:


train_df = pd.read_csv(input_directory + "train.csv")
train_df.rename(index=str, columns={"label": "target"}, inplace=True)
train_df.head()


# In[ ]:


test_df = pd.read_csv(input_directory + "test.csv")
test_df.rename(index=str, columns={"label": "target"}, inplace=True)
test_df.head()


# # 4. Visualization

# In[ ]:


ticksize = 18
titlesize = ticksize + 8
labelsize = ticksize + 5

figsize = (18, 5)
params = {'figure.figsize' : figsize,
          'axes.labelsize' : labelsize,
          'axes.titlesize' : titlesize,
          'xtick.labelsize': ticksize,
          'ytick.labelsize': ticksize}

plt.rcParams.update(params)

col = "target"
xlabel = "Label"
ylabel = "Count"

sns.countplot(x=train_df[col])
plt.title("Label Count")
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.show()


# # 5. Preprocess

# In[ ]:


def get_data(train_X=None, train_Y=None, test_X=None, batch_size=32):
    print("Preprocessing and Generating Data Batches.......\n")
    
    rescale = 1.0/255

    train_batch_size = batch_size
    validation_batch_size = batch_size*5
    test_batch_size = batch_size*5
    
    train_shuffle = True
    val_shuffle = True
    test_shuffle = False
    
    train_datagen = ImageDataGenerator(
        horizontal_flip=False,
        vertical_flip=False,
        rotation_range=10,
#         shear_range=15,
        zoom_range = 0.1, 
        width_shift_range=0.1,  
        height_shift_range=0.1,
        rescale=rescale,
        validation_split=0.25)

    
    train_generator = train_datagen.flow(
        x=train_X, 
        y=train_Y, 
        batch_size=batch_size,
        shuffle=True, 
        sample_weight=None, 
        seed=42, 
        save_to_dir=None, 
        save_prefix='', 
        save_format='png', 
        subset='training')
    
    
    validation_generator = train_datagen.flow(
        x=train_X, 
        y=train_Y, 
        batch_size=validation_batch_size,
        shuffle=True, 
        sample_weight=None, 
        seed=42, 
        save_to_dir=None, 
        save_prefix='', 
        save_format='png', 
        subset='validation')
    
    test_datagen = ImageDataGenerator(rescale=rescale)
    
    test_generator = test_datagen.flow(
        x=test_X, 
        y=None,  
        batch_size=test_batch_size,
        shuffle=False, 
        sample_weight=None, 
        seed=42, 
        save_to_dir=None, 
        save_prefix='', 
        save_format='png')
    
    class_weights = get_weight(np.argmax(train_Y, axis=1))
    
    steps_per_epoch = len(train_generator)
    validation_steps = len(validation_generator)
    
    print("\nPreprocessing and Data Batch Generation Completed.\n")
    
    
    return train_generator, validation_generator, test_generator, class_weights, steps_per_epoch, validation_steps
            
# Calculate Class Weights
def get_weight(y):
    class_weight_current =  cw.compute_class_weight('balanced', np.unique(y), y)
    return class_weight_current


# # 5. Model Function

# In[ ]:


def get_model(model_name, input_shape=(96, 96, 3), num_class=2, weights='imagenet', dense_units=1024, internet=False):
    inputs = Input(input_shape)
    
    if model_name == "Xception":
        base_model = Xception(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "ResNet50":
        base_model = ResNet50(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "ResNet101":
        base_model = keras.applications.resnet.ResNet101(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "ResNet152":
        base_model = keras.applications.resnet.ResNet152(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "ResNet50V2":
        base_model = resnet_v2.ResNet50V2(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "ResNet101V2":
        base_model = resnet_v2.ResNet101V2(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "ResNet152V2":
        base_model = resnet_v2.ResNet152V2(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "ResNeXt50":
        base_model = resnext.ResNeXt50(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "ResNeXt101":
        base_model = resnext.ResNeXt101(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "InceptionV3":
        base_model = InceptionV3(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "InceptionResNetV2":
        base_model = InceptionResNetV2(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "DenseNet201":
        base_model = DenseNet201(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "NASNetMobile":
        base_model = NASNetMobile(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "NASNetLarge":
        base_model = NASNetLarge(include_top=False, weights=weights, input_shape=input_shape)
        
        
#     x = base_model(inputs)
#     x = Dropout(0.5)(x)
    
#     out1 = GlobalMaxPooling2D()(x)
#     out2 = GlobalAveragePooling2D()(x)
#     out3 = Flatten()(x)
    
#     out = Concatenate(axis=-1)([out1, out2, out3])
    
#     out = Dropout(0.6)(out)
#     out = BatchNormalization()(out)
#     out = Dropout(0.5)(out)
    
#     if num_class>1:
#         out = Dense(num_class, activation="softmax", name="3_")(out)
#     else:
#         out = Dense(1, activation="sigmoid", name="3_")(out)
        
#     model = Model(inputs, out)
#     model = Model(inputs=base_model.input, outputs=outputs)

    
    x = base_model.output
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(dense_units)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.5)(x)
    
    if num_class>1:
        outputs = Dense(num_class, activation="softmax")(x)
    else:
        outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    
    model.summary()
    
    
    return model


def get_conv_model(num_class=2, input_shape=None, dense_units=256):
    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation ='relu', input_shape = input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation ='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))


    model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation ='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation ='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))

    
    model.add(Flatten())
    model.add(Dense(dense_units, activation = "relu"))
    model.add(Dropout(0.5))

    
#     model.add(Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4), input_shape = input_shape))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
    
#     model.add(Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
    
#     model.add(MaxPool2D())
#     model.add(Dropout(0.5))

    
#     model.add(Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
    
#     model.add(Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
    
#     model.add(MaxPool2D())
#     model.add(Dropout(0.5))
    
    
#     model.add(GlobalAveragePooling2D())
    
    
    if num_class>1:
        model.add(Dense(num_class, activation='softmax'))
    else:
        model.add(Dense(num_class, activation='sigmoid'))
    
    print(model.summary())

    return model


# ## Visualization

# In[ ]:


def plot_performance(history=None, figure_directory=None):
    xlabel = 'Epoch'
    legends = ['Training', 'Validation']

    ylim_pad = [0.005, 0.005]
    ylim_pad = [0, 0]


    plt.figure(figsize=(20, 5))

    # Plot training & validation Accuracy values

    y1 = history.history['accuracy']
    y2 = history.history['val_accuracy']

    min_y = min(min(y1), min(y2))-ylim_pad[0]
    max_y = max(max(y1), max(y2))+ylim_pad[0]
    
#     min_y = .96
#     max_y = 1


    plt.subplot(121)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Accuracy\n'+date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()


    # Plot training & validation loss values

    y1 = history.history['loss']
    y2 = history.history['val_loss']

    min_y = min(min(y1), min(y2))-ylim_pad[1]
    max_y = max(max(y1), max(y2))+ylim_pad[1]

#     min_y = .1
#     max_y = 0

    plt.subplot(122)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Loss\n'+date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()
    if figure_directory:
        plt.savefig(figure_directory+"/history")

    plt.show()


# # 6. Output Configuration

# In[ ]:


main_model_dir = output_directory + r"models_output/"
main_log_dir = output_directory + r"logs/"

try:
    os.mkdir(main_model_dir)
except:
    print("Could not create main model directory")
    
try:
    os.mkdir(main_log_dir)
except:
    print("Could not create main log directory")



model_dir = main_model_dir + time.strftime('%Y-%m-%d %H-%M-%S') + "/"
log_dir = main_log_dir + time.strftime('%Y-%m-%d %H-%M-%S')


try:
    os.mkdir(model_dir)
except:
    print("Could not create model directory")
    
try:
    os.mkdir(log_dir)
except:
    print("Could not create log directory")
    
model_file = model_dir + "{epoch:02d}-val_acc-{val_acc:.2f}-val_loss-{val_loss:.2f}.hdf5"


# ## 6.2 Call Back Configuration

# In[ ]:


print("Settting Callbacks")

def step_decay(epoch, lr):
    # initial_lrate = 1.0 # no longer needed
    lrate = lr
    if epoch==2:
        lrate = 0.0001  
#     lrate = lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


checkpoint = ModelCheckpoint(
    model_file, 
    monitor='val_acc', 
    save_best_only=True)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True)


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.6,
    patience=2,
    min_lr=0.0000001,
    verbose=1)

learning_rate_scheduler = LearningRateScheduler(step_decay, verbose=1)
# f1_metrics = Metrics()


callbacks = [reduce_lr, early_stopping]
# callbacks = [checkpoint, reduce_lr, early_stopping]
# callbacks = [reduce_lr, early_stopping, f1_metrics]

print("Set Callbacks at ", date_time(1))


# # 7. Model

# In[ ]:


print("Getting Base Model", date_time(1))

# model_name="InceptionV3"
# model_name="NASNetMobile"

dim = 28

input_shape = (dim, dim, 1)


num_class = len(set(train_df["target"].values))

weights = 'imagenet'
dense_units = 256

internet = True

# model = get_model(model_name=model_name, 
#                   input_shape=input_shape, 
#                   num_class=num_class, 
#                   weights=weights, 
#                   dense_units=dense_units, 
#                   internet=internet)

model = get_conv_model(num_class=num_class, input_shape=input_shape, dense_units=dense_units)
print("Loaded Base Model", date_time(1))


# In[ ]:


loss = 'categorical_crossentropy'
# loss = 'binary_crossentropy'
metrics = ['accuracy']
# metrics = [auroc]


# # 8. Data

# In[ ]:


# train_X = train_df.drop(columns=["target"]).values
# train_Y = train_df["target"].values


# clf = svm.SVC()

# cross_val_score(clf, train_X, train_Y, cv=10, n_jobs=-1, verbose=2)


# In[ ]:


train_X = train_df.drop(columns=["target"]).values
train_X = train_X.reshape(train_X.shape[0], dim, dim,1)

train_Y = train_df["target"].values
train_Y = keras.utils.to_categorical(train_Y, 10) 

test_X = test_df.drop(columns=["id"]).values
test_X = test_X.reshape(test_X.shape[0], dim, dim,1)


# In[ ]:


batch_size = 128

# class_mode = "categorical"
# class_mode = "binary"

# target_size = (dim, dim)

train_generator, validation_generator, test_generator, class_weights, steps_per_epoch, validation_steps = get_data(train_X=train_X, train_Y=train_Y, test_X=test_X, batch_size=batch_size)


# # 9. Training

# In[ ]:


print("Starting Trainning ...\n")

start_time = time.time()
print(date_time(1))

# batch_size = 32
# train_generator, validation_generator, test_generator, class_weights, steps_per_epoch, validation_steps = get_data(batch_size=batch_size)

print("\n\nCompliling Model ...\n")
learning_rate = 0.001
optimizer = Adam(learning_rate)
# optimizer = Adam()

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

steps_per_epoch = len(train_generator)
validation_steps = len(validation_generator)

verbose = 1
epochs = 100

print("Trainning Model ...\n")
history = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    verbose=verbose,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=validation_steps, 
    class_weight=class_weights)

elapsed_time = time.time() - start_time
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

print("\nElapsed Time: " + elapsed_time)
print("Completed Model Trainning", date_time(1))


# # 10. Model Performance 
# Model Performance  Visualization over the Epochs

# In[ ]:


plot_performance(history=history)


# In[ ]:


ypreds = model.predict_generator(generator=test_generator, steps = len(test_generator),  verbose=1)
# ypreds


# In[ ]:


# ypred = ypreds[:,1]#
ypred = np.argmax(ypreds, axis=1)


# In[ ]:


sample_df = pd.read_csv(input_directory+"sample_submission.csv")
sample_df.head()


# In[ ]:


test_gen_id = test_generator.index_array
sample_submission_id = sample_df["id"]

len(test_gen_id), len(sample_submission_id)


# In[ ]:


sample_list = list(sample_df.id)

pred_dict = dict((key, value) for (key, value) in zip(test_generator.index_array, ypred))

pred_list_new = [pred_dict[f] for f in sample_list]

test_df = pd.DataFrame({'id': sample_list,'label': pred_list_new})

test_df.to_csv('submission.csv', header=True, index=False)


# In[ ]:


test_df.head()


# In[ ]:




