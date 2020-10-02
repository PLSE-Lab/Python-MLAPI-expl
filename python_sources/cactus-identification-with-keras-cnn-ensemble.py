#!/usr/bin/env python
# coding: utf-8

# This Kernel deals with the Aerial Cactus (32 x 32) images . It is a good candidate for initial learning . With CNN and basic image preprocessing I was getting good score . Now I have added simple three model ensemble to boost the score . 
# 
# We can achieve better result by directly using fast.ai or transfer learning , however I wanted to start with simple Keras CNN.
# 
# Thanks to the kernel "CNN using Keras" from user @Anirudh Chakravarthy to get me started
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import cv2
from tqdm import tqdm, tqdm_notebook
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,DepthwiseConv2D, Dense, Flatten, Dropout, BatchNormalization, LeakyReLU,GlobalAveragePooling2D, Activation, Average,AveragePooling2D

from keras.optimizers import Adam,Adamax
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

seed = 42
np.random.seed(seed)


# ### Set train and test directories

# In[ ]:


base_dir = os.path.join("..", "input") # set base directory
train_df = pd.read_csv(os.path.join(base_dir, "train.csv"))
train_dir = os.path.join(base_dir, "train/train")
test_dir = os.path.join(base_dir, "test/test")

# print(os.listdir(train_dir))


# ### Using Image Generators for preprocessing input images

# Image Generators have been used to augment the existing data. Training set is split in a 80:20 into train and validation set. Generators are created for each split. 

# In[ ]:


train_df['has_cactus'] = train_df['has_cactus'].astype(str)

batch_size = 64
train_size = 15750
validation_size = 1750

datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.1,
    #zoom_range =0.3,
    zca_whitening = False
    )

data_args = {
    "dataframe": train_df,
    "directory": train_dir,
    "x_col": 'id',
    "y_col": 'has_cactus',
    "featurewise_center" : True,
    "featurewise_std_normalization" : True,
    "samplewise_std_normalization" : False,
    "samplewise_center" : False,
    "shuffle": True,
    "target_size": (32, 32),
    "batch_size": batch_size,
    "class_mode": 'binary'
}

train_generator = datagen.flow_from_dataframe(**data_args, subset='training')
validation_generator = datagen.flow_from_dataframe(**data_args, subset='validation')


# In[ ]:


train_df.head(1)


# In[ ]:


def show_image():
    f, ax = plt.subplots(1,10,figsize=(15,15))
    for i in range(10):
        image1 = next(train_generator)
        img1 = array_to_img(image1[0][0])
        ax[i].imshow(img1)
        ax[i].axis("off")
        
    plt.show()
    plt.axis("off")
    plt.title("Preprocessed", fontsize=18)


#image2 = next(train_generator)
show_image()


# ### Build the model
# I have used three models Le_Conv , Custom_Conv and Complex_model  and trained the models individually before using the predictions from the model . All models use 50 Epochs and early stopping . Training the model happens in a simple for loop

# In[ ]:


def Le_Conv():
    model1 = Sequential()
    model1.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(32, 32, 3), padding="same"))
    model1.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
    model1.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
    model1.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model1.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
    model1.add(Flatten())
    model1.add(Dense(84, activation='relu'))
    model1.add(Dense(1, activation='sigmoid'))
    model1.compile(optimizer =  Adamax(lr =0.001) , loss = "binary_crossentropy", metrics=["acc"])
    return model1

def VggNet_Model():
    model = Sequential()
    model.add(Conv2D(16, (2,2), activation="relu", input_shape=(32, 32, 3)))
    #model.add(LeakyReLU(alpha =0.3))
    model.add(Conv2D(16, (2,2), activation="relu"))
    model.add(BatchNormalization())
    #model.add(LeakyReLU(alpha =0.3))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.3))
    model.add(Conv2D(32, (3,3), activation="relu"))
    #model.add(LeakyReLU(alpha =0.3))
    model.add(Conv2D(32, (3,3), activation="relu"))
    model.add(BatchNormalization())
    #model.add(LeakyReLU(alpha =0.3))
    #model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3,3), activation="relu"))
    #model.add(LeakyReLU(alpha =0.3))
    model.add(Conv2D(64, (3,3), activation="relu"))
    model.add(BatchNormalization())
    #model.add(LeakyReLU(alpha =0.3))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.3))
    #model.add(Conv2D(128, (3,3), activation="relu"))
    #model.add(BatchNormalization())
    #model.add(LeakyReLU(alpha =0.3))
    #model.add(Conv2D(128, (3,3), activation="relu"))
    #model.add(BatchNormalization())
    #model.add(LeakyReLU(alpha =0.3))
    #model.add(MaxPooling2D(2,2))
    #model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(units=1, activation='sigmoid'))
    model.summary()
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, 
                 loss='binary_crossentropy',
                 metrics=['acc'])
    
    return model

def Custom_Conv():
    model = Sequential()
    model.add(Conv2D(32, (5,5), activation="relu", input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
#model.add(LeakyReLU(alpha =0.3))
    model.add(Conv2D(32, (5,5), activation="relu"))
    model.add(BatchNormalization())
#model.add(LeakyReLU(alpha =0.3))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3,3), activation="relu"))
    model.add(BatchNormalization())
#model.add(LeakyReLU(alpha =0.3))
    model.add(Conv2D(128, (3,3), activation="relu"))
    model.add(BatchNormalization())
#model.add(LeakyReLU(alpha =0.3))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units=1, activation='sigmoid'))
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, 
                 loss='binary_crossentropy',
                 metrics=['acc'])
    return model

def Complex_model():
    model = Sequential()
        
    model.add(Conv2D(3, kernel_size = 3, activation = 'relu', input_shape = (32, 32, 3)))
    
    model.add(Conv2D(filters = 16, kernel_size = 3, activation = 'relu'))
    model.add(Conv2D(filters = 16, kernel_size = 3, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))
    model.add(Conv2D(filters = 32, kernel_size = 1, activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = 1, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(DepthwiseConv2D(kernel_size = 3, strides = 2, padding = 'Same', use_bias = True))
    model.add(Conv2D(filters = 128, kernel_size = 1, activation = 'relu'))
    model.add(Conv2D(filters = 256, kernel_size = 1, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))
    model.add(Conv2D(filters = 256, kernel_size = 1, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 512, kernel_size = 1, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(DepthwiseConv2D(kernel_size = 3, strides = 2, padding = 'Same', use_bias = True))
    model.add(Conv2D(filters = 512, kernel_size = 1, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 1024, kernel_size = 1, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))
    model.add(Conv2D(filters = 1024, kernel_size = 1, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 2048, kernel_size = 1, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    #model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    
    model.add(Dense(470, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(128, activation = 'tanh'))

    model.add(Dense(1, activation = 'sigmoid'))
    
    opt=Adam(0.001)
    model.compile(optimizer = opt, loss = 'mean_squared_error', metrics = ['accuracy'])
    
    return model


# ##### Creating a list of models for looping through

# In[ ]:


model = []
#model.append(VggNet_Model())
model.append(Custom_Conv())
model.append(Le_Conv())
model.append(Complex_model())


# ### Set callbacks for training

# These are some standard callbacks which keras provides. 
# 1. EarlyStopping: Stops the training process if the monitored parameter stops improving with 'patience' number of epochs.
# 2. ReduceLROnPlateau: Reduces learning rate by a factor if monitored parameter stops improving with 'patience' number of epochs. This helps fit the training data better.
# 3. TensorBoard: Helps in visualization.
# 4. ModelCheckpoint: Stores the best weights after each epoch in the path provided.
# 
# For further details, refer [this link.](https://keras.io/callbacks)

# ### Train Models in a loop and store the trained models

# In[ ]:


models = []
histories =[]

for i in range(len(model)):
    ckpt_path = 'aerial_cactus_detection_'+str(i)+'.hdf5'
    earlystop = EarlyStopping(monitor='val_acc', patience=25, verbose=1, restore_best_weights=False)
    reducelr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
    modelckpt_cb = ModelCheckpoint(ckpt_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    tb = TensorBoard()
    callbacks = [earlystop, reducelr, modelckpt_cb, tb]
    history = model[i].fit_generator(train_generator,
              validation_data=validation_generator,
              steps_per_epoch=train_size//batch_size,
              validation_steps=validation_size//batch_size,
              epochs=80, verbose=1, 
              shuffle=True,
              callbacks=callbacks)
    models.append(model[i])
    histories.append(history)


# ### Train vs Validation Visualization
# 
# These plots can help realize cases of overfitting.

# In[ ]:


# Training plots
for history in histories :
    epochs = [i for i in range(1, len(history.history['loss'])+1)]
    plt.plot(epochs, history.history['loss'], color='blue', label="training_loss")
    plt.plot(epochs, history.history['val_loss'], color='red', label="validation_loss")
    plt.legend(loc='best')
    plt.title('loss')
    plt.xlabel('epoch')
    plt.show()
    plt.plot(epochs, history.history['acc'], color='blue', label="training_accuracy")
    plt.plot(epochs, history.history['val_acc'], color='red',label="validation_accuracy")
    plt.legend(loc='best')
    plt.title('accuracy')
    plt.xlabel('epoch')
    plt.show()


# ### Get Test Set images for prediction

# In[ ]:


test_df = pd.read_csv(os.path.join(base_dir, "sample_submission.csv"))
print(test_df.head())
test_images = []
images = test_df['id'].values

for image_id in images:
    test_images.append(cv2.imread(os.path.join(test_dir, image_id)))
    
test_images = np.asarray(test_images)
test_images = test_images / 255.0
print("Number of Test set images: " + str(len(test_images)))


# ### Make predictions on test set

# In[ ]:


final_pred =[]
predictcustom = models[0].predict(test_images)
predictle = models[1].predict(test_images)
predictcomplex = models[2].predict(test_images)
#predict4 = models[3].predict(test_images)
stacked_arrays = np.hstack((predictcustom, predictle, predictcomplex)) #,predict4))
pred =np.mean(stacked_arrays, axis=1)
predweighted = 0.3*predictcustom +0.2*predictle + 0.5*predictcomplex


# In[ ]:


test_df['has_cactus'] = predweighted
#test_df['has_cactus1'] = labels
#test_df['has_cactus2'] = final_predict
test_df.to_csv('aerial-cactus-submission.csv', index = False)


# In[ ]:


test_df.head(5)

