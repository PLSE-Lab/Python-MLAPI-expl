#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import scipy
 
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from scipy import stats
#preprocess.
from keras.preprocessing.image import ImageDataGenerator

#dl libraraies
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau

# specifically for cnn
#from conv.conv import ShallowNet
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
 
import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2 
import h5py
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image

#TL pecific modules
from keras.applications.vgg16 import VGG16


# In[ ]:


# from google.colab import drive
# drive.mount('/content/gdrive')


# In[ ]:


#tf.reset_default_graph()


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


def loadDataH5():
    with h5py.File('/kaggle/input/data1h5/data1.h5','r') as hf:
        trainX = np.array(hf.get('trainX'))
        trainY = np.array(hf.get('trainY'))
        valX = np.array(hf.get('valX'))
        valY = np.array(hf.get('valY'))
        print (trainX.shape,trainY.shape)
        print (valX.shape,valY.shape)
    return trainX, trainY, valX, valY


# In[ ]:


trainX, trainY, testX, testY = loadDataH5()


# In[ ]:


type(trainX)


# In[ ]:


# flower17 class names
class_names = ["daffodil", "snowdrop", "lilyvalley", "bluebell", "crocus",
			   "iris", "tigerlily", "tulip", "fritillary", "sunflower", 
			   "daisy", "coltsfoot", "dandelion", "cowslip", "buttercup",
			   "windflower", "pansy"]


# In[ ]:


# np.random.seed(42)
# rn.seed(42)
# tf.set_random_seed(42)


# **PART A**
# 
# Part A requires you to build a range of convolutional networks for tackling the Flowers dataset problem. It also requires you to explore the impact of data augmentation and investigate an ensemble technique.

# In[ ]:


#We are going to initialize batch size and the number of epochs which is going
# to be used across the code
batch_size = 32
epochs=50


# **PART A : 1**
# 
# **Following are the models which are going to be implemented in this part of the assignment**
# 1. Single CNN model with single pooling layer
# 2. First variant of CNN with 2 CNN layers
# 3. Second variant with increased density of each CNN layer
# 
# 

# In[ ]:


#tf.reset_default_graph()


# **PART A : 1**
# 
# **Implementing a baseline CNN, which contains just a single convolutional layer and a single pooling layer.**

# In[ ]:


def singleCNN(width, height, depth, classes):
    # initialize the model along with the input shape to be "channels last"
    model = tf.keras.Sequential() 
    inputShape = (height, width, depth)

    # define the first (and only) CONV => RELU layer
    model.add(tf.keras.layers.Conv2D (64, (3, 3), padding="same", input_shape=inputShape, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # softmax classifier
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(classes, activation='softmax'))
  
    return model


# In[ ]:


# initialize the optimizer and model
print("Compiling model...")

opt = tf.keras.optimizers.SGD(lr=0.01)
model = singleCNN(width=128, height=128, depth=3, classes=17)
print (model.summary())

model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("Training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),	batch_size=32, epochs=epochs)


# In[ ]:


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 50), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 50), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


# **PART A : 1**
# 
# **First CNN Variant by adding 2 CNN layers**

# In[ ]:


# #Declare this as global:
# global graph
# graph = tf.get_default_graph()
#tf.reset_default_graph()


# In[ ]:


def firstCNNVariant(width, height, depth, classes):
  
    # initialize the model along with the input shape
    model1 = tf.keras.Sequential()
    inputShape = (height, width, depth)
    # first set of CONV => RELU => POOL layers
    model1.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=inputShape, activation='relu'))
    model1.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # second set of CONV => RELU => POOL layers
    model1.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    model1.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # first (and only) set of FC => RELU layers
    model1.add(tf.keras.layers.Flatten())
    model1.add(tf.keras.layers.Dense(500, activation='relu'))
    # softmaxclassifier
    model1.add(tf.keras.layers.Dense(classes, activation='softmax'))
     
    return model1


# In[ ]:


from keras import backend as K
# initialize the optimizer and model
print("Compiling model...")
# with graph.as_default():
opt = tf.keras.optimizers.SGD(lr=0.01)
#opt = SGD(lr = 0.01)
model1 = firstCNNVariant(width=128, height=128, depth=3, classes=17)
print (model1.summary())

model1.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("Training network...")
H1 = model1.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=epochs, verbose=1)


# In[ ]:


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), H1.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), H1.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 50), H1.history["acc"], label="train_acc")
plt.plot(np.arange(0, 50), H1.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


# **PART A : 1**
# 
# **Second variant of CNN : Increasing the density of each CNN layer by adding 2 CNN layers**

# In[ ]:


#tf.reset_default_graph()


# In[ ]:


def SecondCNNVariant(width, height, depth, classes):
    # initialize the model along with the input shape
    model2 = tf.keras.Sequential()
    inputShape = (height, width, depth)
    # first CONV => CONV => POOL layer set
    model2.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=inputShape, activation='relu'))
    model2.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same",activation='relu'))
    model2.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # second CONV => CONV => POOL layer set
    model2.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same",activation='relu'))
    model2.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same",activation='relu'))
    model2.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # first (and only) set of FC => RELU layers
    model2.add(tf.keras.layers.Flatten())
    model2.add(tf.keras.layers.Dense(512,activation='relu'))
    # softmaxclassifier
    model2.add(tf.keras.layers.Dense(classes, activation='softmax'))

    return model2


# In[ ]:


# initialize the optimizer and model

print("Compiling model...")

opt = tf.keras.optimizers.SGD(lr=0.01)
#opt = SGD(lr = 0.01)
model2 = SecondCNNVariant(width=128, height=128, depth=3, classes=17)
print (model2.summary())

model2.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("Training network...")
H2 = model2.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=epochs, verbose=1)


# In[ ]:


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), H2.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), H2.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 50), H2.history["acc"], label="train_acc")
plt.plot(np.arange(0, 50), H2.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


# In[ ]:


def thirdCNNVariant(width, height, depth, classes):
    # initialize the model along with the input shape
    model45 = tf.keras.Sequential()
    inputShape = (height, width, depth)
    # first set of CONV => RELU => POOL layers
    model45.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=inputShape, activation='relu'))
    model45.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # second set of CONV => RELU => POOL layers
    model45.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    model45.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # third set of CONV => RELU => POOL layers
    model45.add(tf.keras.layers.Conv2D(96, (3, 3), padding="same", activation='relu'))
    model45.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # first (and only) set of FC => RELU layers
    model45.add(tf.keras.layers.Flatten())
    model45.add(tf.keras.layers.Dense(500, activation='relu'))
    # softmaxclassifier
    model45.add(tf.keras.layers.Dense(classes, activation='softmax'))
    return model45


# **Performing data augmentation**

# In[ ]:


# le=LabelEncoder()
# Y=le.fit_transform(trainY)
# Y=to_categorical(Y,17)


# In[ ]:





# **PART A : 2**
# 
# **We are now going to perform data augmentations and check the impact of it we build in the first part**

# **CNN baseline model with single CNN layer**
# 
# **Data Augmentation with the below mentioned configuration**
# 1. Configuration 1 : shifting the width and height
# 2. Configuration 2 : rotating the image, zooming it and flipping it horizantally
# 3. Configuration 3 : Shear_range, zooming, rotating image, flipping both vertical and horizontal
# 4. Configuration 4 : rotation, zooming and vertical flipping

# **Configuration 1**

# In[ ]:


#Construct the image generator for data augmentation
datagen1 = ImageDataGenerator(
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        )

datagen1.fit(trainX)
datagen1.fit(testX)


# **Configuration 2**

# In[ ]:


#Construct the image generator for data augmentation

datagen2 = ImageDataGenerator(
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        horizontal_flip=True,  # randomly flip images
        )


datagen2.fit(trainX)
datagen2.fit(testX)


# **Configuration 3**

# In[ ]:


#Construct the image generator for data augmentation

datagen3 = ImageDataGenerator(
        shear_range = 0.2,
        zoom_range = 0.2,
        rotation_range = 30,
        horizontal_flip = True,
        vertical_flip = True
        )  


datagen3.fit(trainX)
datagen3.fit(testX)


# **Configuration 4**

# In[ ]:


#Construct the image generator for data augmentation

datagen4 = ImageDataGenerator(
        
        rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.4, # Randomly zoom image 
        horizontal_flip = False,
        vertical_flip=True)  # randomly flip images


datagen4.fit(trainX)
datagen4.fit(testX)


# In[ ]:


# initialize the optimizer and model

print("Compiling model...")

opt = tf.keras.optimizers.SGD(lr=0.01)
#opt = SGD(lr = 0.01)
model = singleCNN(width=128, height=128, depth=3, classes=17)
print (model.summary())

model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("Training network...")
H3 = model.fit_generator(datagen1.flow(trainX,trainY, batch_size=batch_size),
                              epochs = epochs, validation_data = datagen1.flow(testX,testY, batch_size=batch_size),
                              verbose = 1, steps_per_epoch=trainX.shape[0] // batch_size, validation_steps = testX.shape[0] //batch_size)


# In[ ]:





# In[ ]:


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), H3.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), H3.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 50), H3.history["acc"], label="train_acc")
plt.plot(np.arange(0, 50), H3.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


# In[ ]:


# initialize the optimizer and model

print("Compiling model...")

opt = tf.keras.optimizers.SGD(lr=0.01)
#opt = SGD(lr = 0.01)
model = singleCNN(width=128, height=128, depth=3, classes=17)
print (model.summary())

model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("Training network...")
H4 = model.fit_generator(datagen2.flow(trainX,trainY, batch_size=batch_size),
                              epochs = epochs, validation_data = datagen2.flow(testX,testY, batch_size=batch_size),
                              verbose = 1, steps_per_epoch=trainX.shape[0] // batch_size, validation_steps = testX.shape[0] //batch_size)


# In[ ]:


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), H4.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), H4.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 50), H4.history["acc"], label="train_acc")
plt.plot(np.arange(0, 50), H4.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


# In[ ]:


# initialize the optimizer and model

print("Compiling model...")

opt = tf.keras.optimizers.SGD(lr=0.01)
#opt = SGD(lr = 0.01)
model = singleCNN(width=128, height=128, depth=3, classes=17)
print (model.summary())

model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("Training network...")
H5 = model.fit_generator(datagen3.flow(trainX,trainY, batch_size=batch_size),
                              epochs = epochs, validation_data = datagen3.flow(testX,testY, batch_size=batch_size),
                              verbose = 1, steps_per_epoch=trainX.shape[0] // batch_size, validation_steps = testX.shape[0] //batch_size)


# In[ ]:


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), H5.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), H5.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 50), H5.history["acc"], label="train_acc")
plt.plot(np.arange(0, 50), H5.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


# In[ ]:


# initialize the optimizer and model

print("Compiling model...")

opt = tf.keras.optimizers.SGD(lr=0.01)
#opt = SGD(lr = 0.01)
model = singleCNN(width=128, height=128, depth=3, classes=17)
print (model.summary())

model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("Training network...")
H6 = model.fit_generator(datagen4.flow(trainX,trainY, batch_size=batch_size),
                              epochs = epochs, validation_data = datagen4.flow(testX,testY, batch_size=batch_size),
                              verbose = 1, steps_per_epoch=trainX.shape[0] // batch_size, validation_steps = testX.shape[0] //batch_size)


# In[ ]:


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), H6.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), H6.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 50), H6.history["acc"], label="train_acc")
plt.plot(np.arange(0, 50), H6.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


# **Data augmentation Conclusion**
# 
# **The following is the accuracy obtained by performing data augmentation using 4 different configuration**
# 
#         Configuration    Accuracy
#         Config1          60.59
#         Config2          63.24
#         Config3          67.65
#         Config4          61.18
#  **We can see that in configuration 3 we obtained the highest accuracy and hence we can conclude that data augmentation helped us to increase the accuracy of the model.**
#  
#  **Note: The accuracy might change very slightly for different runs**

# **PART A: 3**
# 
# **Ensemble CNN**
# 
# 
# **The goal of this experiment is to build an CNN Ensemble model with 5 base learners. We are performing Ensemble by two different ways**
# 
# 1. We use different networks and train them to same number of epochs and with same batch size. We are also keeping the learning rate same.
# 2. We train one single network with fixed structure for multiple times( Hre we are training for 20 times).
# 
# **In both the methods, we compare the validation accuracy of each base learner and try to see if we can achieve a better validation accuracy**
# 
# **We perform two ensemble methods**
# 1. Ensemble by averaging the predictions(Method of aggregation)
# 2. Ensemble by  variablity in the base learner

# **We first try the ensemble method by variability of base learners**
# **Here we are considering 5 different neural network models with different archiectures and then we are aggregating there results by averaging out the predcitions and also used another method called voting. These two ways can help us determine the final accuracy of the model.**

# In[ ]:


def fourthCNNVariant(width, height, depth, classes):
    # initialize the model along with the input shape
    model46 = tf.keras.Sequential()
    inputShape = (height, width, depth)
    # first set of CONV => RELU => POOL layers
    model46.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=inputShape, activation='relu'))
    model46.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # second set of CONV => RELU => POOL layers
    model46.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    model46.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # third set of CONV => RELU => POOL layers
    model46.add(tf.keras.layers.Conv2D(96, (3, 3), padding="same", activation='relu'))
    model46.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # forth set of CONV => RELU => POOL layers
    model46.add(tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'))
    model46.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # first (and only) set of FC => RELU layers
    model46.add(tf.keras.layers.Flatten())
    model46.add(tf.keras.layers.Dense(500, activation='relu'))
    # softmaxclassifier
    model46.add(tf.keras.layers.Dense(classes, activation='softmax'))
    return model46


# In[ ]:


def fifthCNNVariant(width, height, depth, classes):
    # initialize the model along with the input shape
    model48 = tf.keras.Sequential()
    inputShape = (height, width, depth)
    # first set of CONV => RELU => POOL layers
    model48.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=inputShape, activation='relu'))
    model48.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # second set of CONV => RELU => POOL layers
    model48.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    model48.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # third set of CONV => RELU => POOL layers
    model48.add(tf.keras.layers.Conv2D(96, (3, 3), padding="same", activation='relu'))
    model48.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # forth set of CONV => RELU => POOL layers
    model48.add(tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'))
    model48.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # fifth set of CONV => RELU => POOL layers
    model48.add(tf.keras.layers.Conv2D(160, (3, 3), padding="same", activation='relu'))
    model48.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # first (and only) set of FC => RELU layers
    model48.add(tf.keras.layers.Flatten())
    model48.add(tf.keras.layers.Dense(500, activation='relu'))
    # softmaxclassifier
    model48.add(tf.keras.layers.Dense(classes, activation='softmax'))
    return model48


# In[ ]:


def train_Model(trainX,trainY,testX,testY):
    print("Initializing various models")
    models = []
    models.append(singleCNN(width=128, height=128, depth=3, classes=17))
    models.append(firstCNNVariant(width=128, height=128, depth=3, classes=17))
    models.append(thirdCNNVariant(width=128, height=128, depth=3, classes=17))
    models.append(fourthCNNVariant(width=128, height=128, depth=3, classes=17))
    models.append(fifthCNNVariant(width=128, height=128, depth=3, classes=17))
    print("The total number of models", len(models))
    val = []
    for i in np.arange(0,len(models)):
        print("[INFO] training model {}/{}".format(i+1, len(models)))
        
        models[i].compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        H8 = models[i].fit_generator(datagen2.flow(trainX,trainY, batch_size=batch_size),
                              epochs = epochs, validation_data = datagen2.flow(testX,testY, batch_size=batch_size),
                              verbose = 1, steps_per_epoch=trainX.shape[0] // batch_size,)
        


        val.append(models[i])
    # plot the training loss and accuracy
        N = epochs
        p = ['model_{}.png'.format(i)]
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(np.arange(0, N), H8.history['loss'],
                 label='train_loss')
        plt.plot(np.arange(0, N), H8.history['val_loss'],
                 label='val_loss')
        plt.plot(np.arange(0, N), H8.history['acc'],
                 label='train-acc')
        plt.plot(np.arange(0, N), H8.history['val_acc'],
                 label='val-acc')
        plt.title("Training Loss and Accuracy for model {}".format(i))
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        #plt.savefig(os.path.sep.join(p))
        plt.close()
    return val

              


# In[ ]:





# In[ ]:


import scipy
def predict(val , testX, testY):

    # load the trained convolutional neural network
    print("[INFO] loading network...")
    models = val


    labelName = class_names


    print("[INFO] evaluating ensemble...")
    predictions = []
    accuracy_model = []
    
    for model in models:
        
        predictions.append(model.predict(testX,batch_size=64))
        
        print("##############################################################")
    print("[INFO] Ensemble with Averaging")
    
    predictions = np.average(predictions,axis=0)
      
    print("##############################################################")
    print('\n')
    print("[INFO] Ensemble with voting")
    
    labels = []
    for m in models:
        predicts = np.argmax(m.predict(testX, batch_size=64), axis=1)
        labels.append(predicts)
    #print("labels_append:", labels)
    # Ensemble with voting
    labels = np.array(labels)
    #print("labels_array:", labels)
    
    labels = np.transpose(labels, (1, 0))
    #print("labels_transpose:", labels)
        
    labels = scipy.stats.mode(labels, axis=1)[0]
    #print("labels_mode:", labels)
    labels = np.squeeze(labels)
    #print("labels: ", labels)
    print(classification_report(testY,labels, target_names=labelName))
    accu = accuracy_score(testY, labels)
    return accu


# In[ ]:


if __name__ == '__main__':
    val = train_Model(trainX,trainY,testX,testY)
    accuracy_ensemble = predict(val, testX, testY)
    print('The accuracy of the ensemble model obtained is : ', accuracy_ensemble)


# In[ ]:





# **We try the ensemble by training a fixed network for multiple times**
# 
# **This method is called Method of aggregation, where we train the single archiecture of the model for different number of iterations and then average out the predicitions**

# In[ ]:


def fixedLearner(trainX,trainY,testX,testY,width, height, depth, classes):
    opt = tf.keras.optimizers.SGD(lr=0.01)
    # initialize the model along with the input shape
    model = tf.keras.Sequential()
    inputShape = (height, width, depth)
    # first set of CONV => RELU => POOL layers
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=inputShape, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # second set of CONV => RELU => POOL layers
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # first (and only) set of FC => RELU layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(500, activation='relu'))
    # softmaxclassifier
    model.add(tf.keras.layers.Dense(classes, activation='softmax'))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    #fit model
    model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=epochs, verbose=1)
     
    return model


# In[ ]:


def ensembleEvaluation(ensemble_members, testX, testY):
    models = ensemble_members
    predictions = []
    labels = []
    for model in models:
        predictions.append(model.predict(testX,batch_size=64))
    predictions = np.average(predictions,axis=0)
    
    
    for m in models:
        predicts = np.argmax(m.predict(testX, batch_size=64), axis=1)
        labels.append(predicts)
    #print("labels_append:", labels)
    # Ensemble with voting
    labels = np.array(labels)
    #print("labels_array:", labels)
    
    labels = np.transpose(labels, (1, 0))
    #print("labels_transpose:", labels)
        
    labels = scipy.stats.mode(labels, axis=1)[0]
    #print("labels_mode:", labels)
    labels = np.squeeze(labels)
    #print("labels: ", labels)
    #print(classification_report(testY,labels, target_names=labelName))
    result = accuracy_score(testY, labels)
    return predictions, result


# In[ ]:


# initialize the optimizer and model
width=128
height=128
depth=3
classes=17
#Number of times we want our baselearner to run
num_of_iteration = 10

#The number of ensemble member for each run
ensemble_members = [fixedLearner(trainX,trainY,testX,testY,width, height, depth, classes) for _ in range(num_of_iteration) ]

#Individual accuracy score
accuracy_single = []
#Ensemble accuracy score
accuracy_ensemble = []
labels = []
#Calculation of accuracy for each model
  
predictions, accuracy_ensemble = ensembleEvaluation(ensemble_members, testX, testY)


# In[ ]:


#Combined accuracy of the final model
#print('Single accuracy:',accuracy_single)
#print(np.std(accuracy_single))
#print(np.std(predictions))
print('Ensemble accuracy: ',accuracy_ensemble)


# In[ ]:


# # plot score vs number of ensemble members
# x_axis = [i for i in range(1, len(ensemble_members)+1)]
# pyplot.plot(x_axis, accuracy_single, marker='o', linestyle='None')
# pyplot.plot(x_axis, accuracy_ensemble, marker='o')
# pyplot.show()


# **Diversity in ensembles: As we can notice that we tried two ways by which we can inject diversity in ensembles.**
# 1. Variability in base learners
# 2. Method of Aggregation
# and the accuracy of the ensemble model is 
# 
# There is not much of improvement in the overall accuracy of the model. This is due to lack of diversity in the basemodels. But when we compare the model's accuracy we can notice that method of aggregation gave us better results than Variability in base learner.
# 
# Some other ways to improve the accuracy by introducing diversity can be achieved by weighting the importance of each model. By doing this, we can increase the accuracy . Another method is by stacking.
