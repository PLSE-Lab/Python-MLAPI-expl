#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# import libraries
import json
import math
import os
from glob import glob 
from tqdm import tqdm
from PIL import Image
import cv2 # image processing
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # data visualization


from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split

import keras
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.applications import VGG19
from keras.applications import VGG16
from keras.utils.np_utils import to_categorical
from keras.layers import  Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import ImageDataGenerator,image,img_to_array,load_img


# In[ ]:



class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1, n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

        #Own Adds
        self.train_or_test = None
        self.image_folder_path = None

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def set_image_folder_path(self, image_folder_path):
        self.image_folder_path = image_folder_path

    def set_train_or_test(self,train_or_test):
        self.train_or_test = train_or_test

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = self.__data_generation(list_IDs_temp)
        # Generate data
        if self.train_or_test == 'train':
            y = self.__generate_y(list_IDs_temp)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.load_grayscale(ID)

        return X

    def __generate_y(self, list_IDs_temp):

            #Used for binary classificaion if steel is defect or not
        for i, ID in enumerate(list_IDs_temp):
            y = np.empty((self.batch_size), dtype=int)
            y[i] = self.labels[ID]
            y[i] = y[i].astype('float32')
        return y 

    def load_rgb(self, ID):

        img = image.load_img(self.image_folder_path + '\\' + ID + '.jpg' , color_mode = 'rgb', target_size = self.dim)
        img = image.img_to_array(img)
        img = img.astype(np.float32)

        return preprocess_input(img)
    
    def load_grayscale(self,ID):
        img = image.load_img(self.image_folder_path + '\\' + ID + '.jpeg' , color_mode = 'grayscale', target_size = self.dim)
        img = image.img_to_array(img)
        img = img.astype(np.float32)

        return img


# In[ ]:


def picture_ID_generator(path, id_list, target_dict, PNEUMONIA): #Function for appending the names of all files in a folder located at path to an array
    for img in listdir(path):

        id_list.append(img.split('.')[0])
        target_dict[img.split('.')[0]] =  PNEUMONIA
        #id_list.append({'name': img.split('.')[0], 'Pneumonia': PNEUMONIA})


def make_gradcam_heatmap(img_array, model, last_covolutional_layer_name, classification_layer_names):
    #Function for backing out the heatmap (where the CNN looks in the picture)
    #Heavily inspired by https://keras.io/examples/vision/grad_cam/
    
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = Model(model.inputs, last_conv_layer.output)
    classifier_input = keras.Input(shape= last_conv_layer.output.shape[1:])

    x = classifier_input
    for layer in classification_layer_names:
        x = model.get_layer(layer)(x)
    classifier_model = Model(classifier_input, x)

    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        predictions = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(predictions[0])

        top_class_channel = predictions[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis = (0,1,2))    
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:,:,i] *= pooled_grads[i]

    heatmap = np.mean(last_conv_layer_output, axis = -1)

    heatmap = np.maximum(heatmap, 0)/ np.max(heatmap)

    return heatmap


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))


# In[ ]:


import cv2


# In[ ]:



labels = ['PNEUMONIA', 'NORMAL']
img_size = 150
def get_training_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)


# In[ ]:


train = get_training_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/train')
test = get_training_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/test')
val = get_training_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/val')


# In[ ]:


import seaborn as sns


# In[ ]:


l = []
for i in train:
    if(i[1] == 0):
        l.append("Pneumonia")
    else:
        l.append("Normal")
sns.set_style('darkgrid')
sns.countplot(l)        


# In[ ]:


plt.figure(figsize = (5,5))
plt.imshow(train[0][0], cmap='gray')
plt.title(labels[train[0][1]])

plt.figure(figsize = (5,5))
plt.imshow(train[-1][0], cmap='gray')
plt.title(labels[train[-1][1]])


# In[ ]:


x_train = []
y_train = []

x_val = []
y_val = []

x_test = []
y_test = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)
    
for feature, label in val:
    x_val.append(feature)
    y_val.append(label)


# In[ ]:


def normalization_data(data):
    data = np.array(data) / 255
    data = data.reshape(-1, img_size, img_size, 1)
    return data


# In[ ]:


x_train = normalization_data(x_train)
x_test = normalization_data(x_test)
x_val = normalization_data(x_val)


# In[ ]:


y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)


# In[ ]:


np.shape(x_train)


# In[ ]:


input_dir = "../input/chest-xray-pneumonia/chest_xray/"
train_dir = input_dir +"train/"
test_dir = input_dir +"test/"
val_dir = input_dir +"val/"


# In[ ]:



# Data Augmentation
train_datagen = ImageDataGenerator(
      featurewise_center=True,
      samplewise_center=True,
      rescale=1./255,
      rotation_range= 180,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      vertical_flip=True,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')


# In[ ]:


# Create VGG19 Model with Keras library

def build_model(backbone, lr=1e-4):
    model = Sequential()
    model.add(backbone)
    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation="sigmoid"))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=lr),
        metrics=['acc']
    )
    return model

vgg19 = VGG19(
    weights='imagenet',
    include_top=False,
    input_shape=(150,150,3)
)

model = build_model(vgg19 ,lr = 1e-4)
model.summary()


# In[ ]:


history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=16,
    callbacks = [],
    validation_data=test_generator,
    validation_steps=10
)


# In[ ]:


history.history


# In[ ]:


epochs = [i for i in range(16)]  #epoch used in learning
fig_19 , ax_19 = plt.subplots(1,2)
train_acc = history.history['acc']
print(train_acc)
train_loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
fig_19.set_size_inches(20,10)

ax_19[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax_19[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
ax_19[0].set_title('Training & Validation Accuracy')
ax_19[0].legend()
ax_19[0].set_xlabel("Epochs")
ax_19[0].set_ylabel("Accuracy")

ax_19[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax_19[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
ax_19[1].set_title('Testing Accuracy & Loss')
ax_19[1].legend()
ax_19[1].set_xlabel("Epochs")
ax_19[1].set_ylabel("Training & Validation Loss")
plt.show()


# In[ ]:


# Create VGG16 Model with Keras library
from keras.optimizers import Adam, RMSprop
from keras.applications import VGG16

def build_model(backbone, lr=1e-4):
    model = Sequential()
    model.add(backbone)
    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation="sigmoid"))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=lr),
        metrics=['acc']
    )
    return model

vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150,150,3)
)

model = build_model(vgg16 ,lr = 1e-4)
model.summary()


# In[ ]:


history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=16,
    callbacks = [],
    validation_data=test_generator,
    validation_steps=10
)


# In[ ]:


epochs = [i for i in range(16)]  #epoch used in learning
fig_16 , ax_16 = plt.subplots(1,2)
train_acc = history.history['acc']
print(train_acc)
train_loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
fig_16.set_size_inches(20,10)

ax_16[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax_16[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
ax_16[0].set_title('Training & Validation Accuracy')
ax_16[0].legend()
ax_16[0].set_xlabel("Epochs")
ax_16[0].set_ylabel("Accuracy")

ax_16[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax_16[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
ax_16[1].set_title('Testing Accuracy & Loss')
ax_16[1].legend()
ax_16[1].set_xlabel("Epochs")
ax_16[1].set_ylabel("Training & Validation Loss")
plt.show()


# In[ ]:




