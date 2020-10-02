#!/usr/bin/env python
# coding: utf-8

# <p class=MsoNormal><b><span style='font-size:30.0pt;line-height:107%;
# font-family:"Calibri Light",sans-serif'>Convolutional Recurrent Neural Networks for Kannada MNIST</span></b></p>

# <img src="https://miro.medium.com/max/2948/1*etN2RhEkMJrEtJgWLvD9pQ.png" alt="Smiley face" align="center" width="750" height="650">

# <p class=MsoNormal><b><span style='font-size:18.0pt;line-height:107%;
# font-family:"Calibri Light",sans-serif'>What is a CRNN?</span></b></p>
# 
# 
# <p class=MsoNormal><span style='font-size:14.0pt;line-height:107%;font-family:
# "Calibri Light",sans-serif'>A Convolutional Recurrent Neural Network (RCNN) is a combination of a Convolutional Neural Network and Recurrent Neural Network. CRNNs are commonly used in OCR and image segmentation tasks such as license plate recognition. But, in this kernel, we will build a CRNN for image classification. Generally, CRNNs consist of a CNN followed by an RNN. The CNN extracts features from the training data, while the RNN layer splits distributes these features to a LSTM or GRU. </span></p>

# In[ ]:


#Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Dense, Flatten, BatchNormalization, MaxPooling2D, LSTM, GRU, TimeDistributed, Bidirectional
from keras.optimizers import RMSprop
from keras.utils.vis_utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.callbacks import LearningRateScheduler
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split


# In[ ]:


#Initializing the dataframes
sample_submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")
test = pd.read_csv("../input/Kannada-MNIST/test.csv")
train = pd.read_csv("../input/Kannada-MNIST/train.csv")

#Defining x and y from the train dataframe
x = train.drop(["label"],axis=1).values
y = train["label"].values
x_test = test.drop('id', axis=1).values


x = x.reshape(x.shape[0], 28, 28, 1)
y = to_categorical(y, 10)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


# In[ ]:


x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.10, random_state=42) 


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False) 


# In[ ]:


#Define the RMSprop optimizer
opt = RMSprop(learning_rate=0.001,rho=0.9,epsilon=1e-07)


# In[ ]:


model = Sequential()
    
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(1,2), strides=(1,2)))
model.add(Dropout(0.25))
#Start of the RNN part of the network. The features from the CNN are fed to a Bidirectional GRU.
model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(GRU(256,return_sequences=True)))
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(BatchNormalization())
model.add(Dense(128, activation = "relu"))
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.25))
model.add(Dense(10, activation = "softmax"))
          
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


#Visualize the model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# In[ ]:


from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='loss', 
                                            patience=3, 
                                            verbose=0, 
                                            factor=0.4, 
                                            min_lr=0.00001)


# In[ ]:


bs = 512
eps = 45
#Train the model
model.fit_generator(datagen.flow(x_train,y_train, batch_size=bs),
                                        epochs = eps, steps_per_epoch=x_train.shape[0] // bs,
                                        validation_data = (x_valid,y_valid), 
                                        callbacks=[learning_rate_reduction], 
                                        verbose=2,shuffle = True)


# In[ ]:


sample_submission["label"] = model.predict_classes(x_test)
sample_submission.to_csv("submission.csv",index=False)


# In[ ]:


sample_submission.head()

