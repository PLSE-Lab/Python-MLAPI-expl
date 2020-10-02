#!/usr/bin/env python
# coding: utf-8

# # Cat vs. Dog Detection using Deep Learning

# # 1. Import 

# In[ ]:


# System
import sys
import os
import argparse

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
from keras.models import Sequential

# Deep Learning - Keras - Layers
from keras.layers import Convolution1D, concatenate, SpatialDropout1D, GlobalMaxPool1D, GlobalAvgPool1D, Embedding,     Conv2D, SeparableConv1D, Add, BatchNormalization, Activation, GlobalAveragePooling2D, LeakyReLU, Flatten
from keras.layers import Dense, Input, Dropout, MaxPooling2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D,     Lambda, Multiply, LSTM, Bidirectional, PReLU, MaxPooling1D
from keras.layers.pooling import _GlobalPooling1D

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
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
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
print(os.listdir("../input"))


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


input_directory = r"../input/"
output_directory = r"../output/"

training_dir = input_directory + r"train"
testing_dir = input_directory + r"test"

if not os.path.exists(output_directory):
    os.mkdir(output_directory)
    
figure_directory = "../output/figures"
if not os.path.exists(figure_directory):
    os.mkdir(figure_directory)
    
    
file_name_pred_batch = figure_directory+r"/result"
file_name_pred_sample = figure_directory+r"/sample"


# ## 2.1. Label Extraction

# In[ ]:


train_files = os.listdir(training_dir)
test_files = os.listdir(testing_dir)

df_train = pd.DataFrame({"id": train_files})

df_train["label"] = df_train["id"].apply(lambda x:x.split(".")[0])    
df_train.drop(df_train[df_train["label"]=="train"].index, inplace=True)

df_train.head()


# In[ ]:


df_test = pd.DataFrame({"id": test_files})
df_test["label"] = ["cat"]*(len(test_files))
df_test.head()


# In[ ]:


classes = ['cat', 'dog']


# # 3. Visualization

# In[ ]:


def plot_image(file, directory=None, sub=False, aspect=None):
    path = directory + file
    
    img = plt.imread(path)
    
    plt.imshow(img, aspect=aspect)
    plt.title(file)
    plt.xticks([])
    plt.yticks([])
    
    if sub:
        plt.show()
        
def plot_img_df(directory=None, df=None, filename="id", label = "label", count=5):
    label_map = {}
    
    classes = list(set(df[label]))
    
    for l in classes:
        label_map[l] = df[df[label]==l][filename]
        label_map[l] = label_map[l].sample(count, replace=True)
        
    
    
    ncols = 5
    nrows = count//ncols if count%ncols==0 else count//ncols+1
#     print(nrows, ncols)
    
    figsize=(20, ncols*nrows)

    ticksize = 14
    titlesize = ticksize + 8
    labelsize = ticksize + 5


    params = {'figure.figsize' : figsize,
              'axes.labelsize' : labelsize,
              'axes.titlesize' : titlesize,
              'xtick.labelsize': ticksize,
              'ytick.labelsize': ticksize}

    plt.rcParams.update(params)
    
    i=0
    l = 60
    for label in label_map:
        print("{}{}{}".format(" "*l, label.title(), " "*l))
        j=0
        for id, file in label_map[label].iteritems():
            plt.subplot(nrows, 5, j+1)
            plot_image(file, directory, aspect='auto')
            j=j+1
            
        plt.tight_layout()
        plt.show()
        print("\n")
        
        i+=1
        
def plot_img_dir(directory=testing_dir, count=5):
    selected_files = random.sample(os.listdir(testing_dir), count)
    
    ncols = 5
    nrows = count//ncols if count%ncols==0 else count//ncols+1
    
    figsize=(20, ncols*nrows)

    ticksize = 14
    titlesize = ticksize + 8
    labelsize = ticksize + 5


    params = {'figure.figsize' : figsize,
              'axes.labelsize' : labelsize,
              'axes.titlesize' : titlesize,
              'xtick.labelsize': ticksize,
              'ytick.labelsize': ticksize}

    plt.rcParams.update(params)
    
    i=0
    
    for file in selected_files:        
        plt.subplot(nrows, ncols, i+1)
        path = directory + file
        plot_image(file, directory, aspect=aspect)

        i=i+1
    
    plt.tight_layout()
    plt.show()


# In[ ]:


count = 10
print("Sample trainning images")
plot_img_df(directory=training_dir+"/", df=df_train, count=10)


# In[ ]:


count = 10
aspect = 'auto'
print("Sample test images")
plot_img_dir(directory=testing_dir+"/", count=count)


# In[ ]:


figsize=(18, 5)

ticksize = 14
titlesize = ticksize + 8
labelsize = ticksize + 5

xlabel = "Class"
ylabel = "Count"

title = "Class Count"


params = {'figure.figsize' : figsize,
          'axes.labelsize' : labelsize,
          'axes.titlesize' : titlesize,
          'xtick.labelsize': ticksize,
          'ytick.labelsize': ticksize}

plt.rcParams.update(params)

col1 = "negativereason"
col2 = "airline"
sns.countplot(df_train["label"])
plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.plot()


# # 4. Preprocess

# In[ ]:


def get_data(batch_size=32, target_size=(299, 299), class_mode="categorical", training_dir=training_dir, testing_dir=testing_dir):
    print("Preprocessing and Generating Data Batches.......\n")
    
    rescale = 1.0/255

    train_batch_size = batch_size
    test_batch_size = batch_size

    train_shuffle = True
    val_shuffle = True
    test_shuffle = False
    
    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        shear_range=0.2,
        zoom_range=0.2,
        rescale=rescale,
        validation_split=0.25
    )

    train_generator = train_datagen.flow_from_dataframe(
        df_train, 
        training_dir, 
        x_col='id',
        y_col='label', 
        has_ext=True, 
        target_size=target_size, 
        classes = classes,
        class_mode=class_mode, 
        batch_size=batch_size, 
        shuffle=True, 
        seed=42,
        subset='training')
    
    validation_generator = train_datagen.flow_from_dataframe(
        df_train, 
        training_dir, 
        x_col='id',
        y_col='label', 
        has_ext=True, 
        target_size=target_size, 
        classes = classes,
        class_mode=class_mode, 
        batch_size=1024, 
        shuffle=True, 
        seed=42,
        subset='validation')


    test_datagen = ImageDataGenerator(rescale=rescale)
    test_generator = test_datagen.flow_from_dataframe(
        df_test, 
        testing_dir, 
        x_col='id',
        y_col='label', 
        has_ext=True, 
        target_size=target_size, 
        classes = classes,
        class_mode=class_mode, 
        batch_size=1024, 
        shuffle=False
    )
    
    
    class_weights = get_weight(train_generator.classes)
    
    steps_per_epoch = len(train_generator)
    validation_steps = len(validation_generator)
    
    print("\nPreprocessing and Data Batch Generation Completed.\n")
    
    
    return train_generator, validation_generator, test_generator, class_weights, steps_per_epoch, validation_steps

# Calculate Class Weights
def get_weight(y):
    class_weight_current =  cw.compute_class_weight('balanced', np.unique(y), y)
    return class_weight_current


# # 5. Model

# In[ ]:


def get_model(model_name, input_shape=(96, 96, 3), num_class=2):
    inputs = Input(input_shape)
    
    if model_name == "Xception":
        base_model = Xception(include_top=False, input_shape=input_shape)
    elif model_name == "ResNet50":
        base_model = ResNet50(include_top=False, input_shape=input_shape)
    elif model_name == "InceptionV3":
#         base_model = InceptionV3(include_top=False, input_shape=input_shape)
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "InceptionResNetV2":
        base_model = InceptionResNetV2(include_top=False, input_shape=input_shape)
    if model_name == "DenseNet201":
        base_model = DenseNet201(include_top=False, input_shape=input_shape)
    if model_name == "NASNetMobile":
        base_model = NASNetMobile(include_top=False, input_shape=input_shape)
    if model_name == "NASNetLarge":
        base_model = NASNetLarge(include_top=False, input_shape=input_shape)
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    
    predictions = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
#     x = base_model(inputs)
    
#     output1 = GlobalMaxPooling2D()(x)
#     output2 = GlobalAveragePooling2D()(x)
#     output3 = Flatten()(x)
    
#     outputs = Concatenate(axis=-1)([output1, output2, output3])
    
#     outputs = Dropout(0.5)(outputs)
#     outputs = BatchNormalization()(outputs)
    
#     if num_class>1:
#         outputs = Dense(num_class, activation="softmax")(outputs)
#     else:
#         outputs = Dense(1, activation="sigmoid")(outputs)
        
#     model = Model(inputs, outputs)
    
    model.summary()
    
    
    return model

# Custom Convolutional Neural Network 
def get_conv_model(num_class=2, input_shape=(3,150,150)):
    model = Sequential()
    
    model.add(Conv2D(16, (3, 3), activation='relu', padding="same", input_shape=input_shape))
    model.add(Conv2D(16, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Flatten())
    
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    model.add(Dense(num_class , activation='softmax'))

    print(model.summary())
    
    return model


# # 6. Output Configuration

# In[ ]:


main_model_dir = output_directory + r"models/"
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

checkpoint = ModelCheckpoint(
    model_file, 
    monitor='val_acc', 
    save_best_only=True)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    verbose=1,
    restore_best_weights=True)


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.6,
    patience=1,
    verbose=1)

callbacks = [reduce_lr, early_stopping, checkpoint]

callbacks = [checkpoint, reduce_lr, early_stopping]

print("Set Callbacks at ", date_time(1))


# # 7. Load Model 

# In[ ]:


print("Getting Base Model", date_time(1))

# input_shape = (96, 96, 3)
input_shape = (299, 299, 3)

num_class = 2

model = get_model(model_name="InceptionV3", input_shape=input_shape, num_class=num_class)
# model = get_conv_model(input_shape=input_shape)

print("Loaded Base Model", date_time(1))


# In[ ]:


loss = 'categorical_crossentropy'
# loss = 'binary_crossentropy'
metrics = ['acc']
# metrics = [auroc]


# # 8. Data

# In[ ]:


# batch_size = 32
batch_size = 192

class_mode = "categorical"
# class_mode = "binary"

# target_size=(96, 96)
target_size=(299, 299)

train_generator, validation_generator, test_generator, class_weights, steps_per_epoch, validation_steps = get_data(batch_size=batch_size, target_size=target_size, class_mode=class_mode)


# 
# # 9. Training

# #### Trained model on full tranning dataset for 10 epochs and validated on full validation dataset. Adjusted class weight has been used for trainning. For optimization used Adam optimizer with learning rate of 0.0001. For loss calculation used categorical crossentropy and for model performance evaluation used accuracy metrics.  

# In[ ]:


print("Starting Trainning ...\n")

start_time = time.time()
print(date_time(1))


# batch_size = 32
# train_generator, validation_generator, test_generator, class_weights, steps_per_epoch, validation_steps = get_data(batch_size=batch_size)

print("\n\nCompliling Model ...\n")
learning_rate = 0.0001
optimizer = Adam(learning_rate)
# optimizer = Adam()

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# steps_per_epoch = 1
# validation_steps = 1

verbose = 1
epochs = 10

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
print("Elapsed Time/Epoch: " + elapsed_time/epochs)
print("Completed Model Trainning", date_time(1))


# # 10. Model Performance 
# Model Performance  Visualization over the Epochs

# In[ ]:


def plot_performance(history=None, figure_directory=None):
    xlabel = 'Epoch'
    legends = ['Training', 'Validation']

    ylim_pad = [0, 0]


    plt.figure(figsize=(20, 5))

    # Plot training & validation Accuracy values

    y1 = history.history['acc']
    y2 = history.history['val_acc']

    min_y = min(min(y1), min(y2))-ylim_pad[0]
    max_y = max(max(y1), max(y2))+ylim_pad[0]


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


# In[ ]:


plot_performance(history=history)


# In[ ]:


def generate_result(model, test_generator, nsteps=len(test_generator)):
    y_preds = model.predict_generator(test_generator, steps=nsteps, verbose=1) 
    return y_preds, y_preds[:,1], y_preds.argmax(axis=-1)


# In[ ]:


y_preds_all, y_preds, y_preds_class = generate_result(model, test_generator)


# In[ ]:


df_test = pd.DataFrame({"id": test_generator.filenames, "label": y_preds})
df_test['id'] = df_test['id'].apply(lambda x: x.split('.')[0])
df_test['id'] = df_test['id'].astype(int)
df_test = df_test.sort_values('id')
df_test.to_csv('submission.csv', index=False)
df_test.head()


# In[ ]:


label_map = {0: "Cat", 1: "Dog"} 

df_test2 = pd.DataFrame({"id": test_generator.filenames, "label": y_preds_class})

df_test2['id'] = df_test2['id'].apply(lambda x: x.split('.')[0])

df_test2['label'] = df_test2['label'].map(label_map)

df_test2['id'] = df_test2['id'].astype(int)
df_test2 = df_test2.sort_values('id')

df_test2['id'] = df_test2['id'].astype(str)
df_test2['id'] = df_test2['id'].apply(lambda x: x+".jpg")

df_test2.head()


# In[ ]:


n = len(test_files)
f = list(np.arange(1,n))

count = 20
r =random.sample(f, count)

nrows = 4
ncols = 5

fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows*5, ncols*5))    

for i in range(count):
    plt.subplot(4, 5, i+1)
    file = str()
    path = testing_dir+"/" + df_test2['id'][i]
    img = plt.imread(path)
    plt.imshow(img, aspect='auto')
    plt.title(df_test2['label'][i] + "\nFile: " + str(df_test2['id'][i]))
    plt.xticks([])
    plt.yticks([])
    
plt.show()
    


# In[ ]:




