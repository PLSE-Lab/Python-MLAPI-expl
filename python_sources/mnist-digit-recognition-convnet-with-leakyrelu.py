#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Lets read data using pandas `read_csv` method

# In[ ]:


train_data=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_data=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# prints out first five rows of training data

# In[ ]:


train_data.head()


# prints out first five rows of testing data

# In[ ]:


test_data.head()


# Lets see what csv data format we have to submit. visualize sample submission file

# In[ ]:


sample_submission=pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
sample_submission.head()


# Import all dependencies

# In[ ]:


from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten,BatchNormalization,LeakyReLU,ReLU
from tensorflow.keras.models import Sequential,load_model
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import train_test_split


# seperating data into input data and labels. *trainX* variable stores input data and *trainY* contains labels for input images.

# In[ ]:


trainX=train_data.iloc[:,1:].to_numpy()
trainY=train_data.iloc[:,0].to_numpy()
print("train X shape: ", trainX.shape)
print("train Y shape: ", trainY.shape)
testX=test_data.to_numpy()
print("test X shape: ", testX.shape)


# lets format input 784 image pixels into 28X28 image sizes so that we can feed it into convolutional neural network. Also lets change our labels to one hot encoded vectors

# In[ ]:


trainX=trainX.reshape((-1,28,28,1))
trainY=to_categorical(trainY,num_classes = 10)
testX=testX.reshape((-1,28,28,1))
print("train X shape: ", trainX.shape)
print("train Y shape: ", trainY.shape)
print("test X shape: ", testX.shape)


# lets normalize our images pixel values between 0 and 1

# In[ ]:


trainX=trainX/255.0
testX=testX/255.0


# divide data to training and validation sets

# In[ ]:


tX,vX,tY,vY=train_test_split(trainX,trainY,test_size=0.15,stratify=trainY,random_state=2)


# In[ ]:


print("train X shape: ", tX.shape)
print("train Y shape: ", tY.shape)
print("validation X shape: ", vX.shape)
print("validation y shape: ", vY.shape)


# Utitlity function to plot images with true class labels and predicted class labels

# In[ ]:


def plot_images(images, cls_true, cls_pred=None,w=3,h=3):
    assert len(images) == len(cls_true)
    fig, axes = plt.subplots(w, h)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape((28,28)), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


# In[ ]:


plot_images(tX[0:9],np.argmax(tY[0:9],axis=1))


# Define our convolutional neural network

# In[ ]:


model=Sequential()
model.add(Conv2D(input_shape=(28,28,1),filters=64,kernel_size=(5,5),padding='same'))
model.add(Conv2D(filters=64,kernel_size=(5,5),padding='same'))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same'))
model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same'))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same'))
model.add(Conv2D(filters=16,kernel_size=(2,2),padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(LeakyReLU())
model.add(Flatten())
model.add(Dense(1024))
model.add(LeakyReLU())
model.add(Dense(512))
model.add(LeakyReLU())
model.add(Dense(256))
model.add(LeakyReLU())
model.add(Dense(64))
model.add(LeakyReLU())
model.add(Dense(10,activation='softmax'))


# In[ ]:


model.compile(loss="categorical_crossentropy",optimizer='rmsprop',metrics=['accuracy'])


# Image augmentation for reduce model overfitting 

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False, 
        zca_whitening=False,
        rotation_range=10, 
        zoom_range = 0.1, 
        width_shift_range=0.1, 
        height_shift_range=0.1,
        horizontal_flip=False, 
        vertical_flip=False
)


# In[ ]:


datagen.fit(tX)


# In[ ]:


train_datagen=datagen.flow(tX,tY,batch_size=128)


# visulizing the augmented images

# In[ ]:


for i in range(5):
    img=train_datagen.next()
    plot_images(img[0],np.argmax(img[1],axis=1))


# Initializing model callbacks functions

# In[ ]:


checkpointer = ModelCheckpoint(filepath='best_model.h5',monitor="val_accuracy",mode="max", verbose=1, save_best_only=True)
early_stop=EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
reduce_lr=ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)


# lets train our model

# In[ ]:


history=model.fit_generator(train_datagen,steps_per_epoch=len(tX)//128,validation_data=(vX,vY),epochs=200,callbacks=[checkpointer,early_stop,reduce_lr])


# plotting loss and accuracy graph

# In[ ]:


fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# Loading our best model

# In[ ]:


model=load_model('best_model.h5')


# lets predicts values for our test dataset

# In[ ]:


testY=model.predict(testX)
predictions=np.argmax(testY,axis=1)


# creating a *submission,csv* file to submit

# In[ ]:


df=pd.DataFrame()
df['ImageId']=[x for x in range(1,len(testX)+1)]
df['Label'] = predictions
df[['ImageId','Label']].to_csv("submission.csv", index=False)
print(df[['ImageId','Label']].head(20))
plot_images(testX[100:200],np.argmax(testY[100:200],axis=1),w=10,h=10)
print("Done!")
print(testX.shape)


# In[ ]:




