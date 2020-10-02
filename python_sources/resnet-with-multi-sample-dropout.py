#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('ls')


# In[ ]:


import sklearn.datasets
import sklearn.model_selection
import keras.preprocessing.image
import keras.utils
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from skimage import color
from sklearn.metrics import accuracy_score
import keras.callbacks
import os
import numpy as np
import cv2

#def load_data(infDir):
#    infData=sklearn.datasets.load_files(infDir,load_content=False)
#    y_inf = np.array(infData['target'])
#    y_inf_names = np.array(infData['target_names'])
#    nclasses = len(np.unique(y_inf))
#    target_size=50
#    x_inf=[]
#    for filename in infData['filenames']:
#        x_inf.append(
#                keras.preprocessing.image.img_to_array(
#                        keras.preprocessing.image.load_img(filename,target_size=(target_size, target_size))
#                )
#        )
#    return([x_inf,y_inf])
    
    

train_dir = '../input/fruits-360_dataset/fruits-360/Training'
trainData=sklearn.datasets.load_files(train_dir,load_content=False)

test_dir = '../input/fruits-360_dataset/fruits-360/Test'
testData=sklearn.datasets.load_files(test_dir,load_content=False)


y_train = np.array(trainData['target'])
y_train_names = np.array(trainData['target_names'])

y_test = np.array(testData['target'])
y_test_names = np.array(testData['target_names'])

nclasses = len(np.unique(y_train))
target_size=50

x_train=[]
for filename in trainData['filenames']:
    x_train.append(
            keras.preprocessing.image.img_to_array(
                    keras.preprocessing.image.load_img(filename,target_size=(target_size, target_size))
                    )
            )
    
    
x_test=[]
for filename in testData['filenames']:
    x_test.append(
            keras.preprocessing.image.img_to_array(
                    keras.preprocessing.image.load_img(filename,target_size=(target_size, target_size))
                    )
            )


# In[ ]:


x_train=np.array(x_train)
x_train=x_train/255
y_train=keras.utils.np_utils.to_categorical(y_train,nclasses)


x_test=np.array(x_test)
x_test=x_test/255
y_test=keras.utils.np_utils.to_categorical(y_test,nclasses)


# In[ ]:


x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
        x_train, y_train, test_size=0.2
)
print(y_train.shape)
print(y_val.shape)


# In[ ]:


N_SAMPLES = 5

images = keras.layers.Input(x_train.shape[1:])

#inizio blocco 1
x = keras.layers.Conv2D(filters=16, kernel_size=[1, 1], padding='same')(images)
block = keras.layers.Conv2D(filters=16, kernel_size=[3, 3], padding="same")(x)
block = keras.layers.BatchNormalization()(block)
block = keras.layers.Activation("relu")(block)
block = keras.layers.Conv2D(filters=16, kernel_size=[3, 3], padding="same")(block)

# #inio Squeeze and Excitation 1
# sq = keras.layers.GlobalAveragePooling2D()(block)
# sq = keras.layers.Reshape((1,1,16))(sq)
# sq = keras.layers.Dense(units=16,activation="sigmoid")(sq)
# block = keras.layers.multiply([block,sq])
# #fine Squeeze and Excitation 1

net = keras.layers.add([x,block])
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)
net = keras.layers.MaxPooling2D(pool_size=(2, 2),name="block_1")(net)



#fine blocco 1
#inizio blocco 2
x = keras.layers.Conv2D(filters=32, kernel_size=[1, 1], padding='same')(net)
block = keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding="same")(x)
block = keras.layers.BatchNormalization()(block)
block = keras.layers.Activation("relu")(block)
block = keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding="same")(block)

# #inio Squeeze and Excitation 2
# sq = keras.layers.GlobalAveragePooling2D()(block)
# sq = keras.layers.Reshape((1,1,32))(sq)
# sq = keras.layers.Dense(units=32,activation="sigmoid")(sq)
# block = keras.layers.multiply([block,sq])
# #fine Squeeze and Excitation 2


net = keras.layers.add([x,block])
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)
net = keras.layers.MaxPooling2D(pool_size=(2, 2),name="block_2")(net)
#fine blocco 2
#inizio blocco 3
x = keras.layers.Conv2D(filters=64, kernel_size=[1, 1], padding='same')(net)
block = keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding="same")(x)
block = keras.layers.BatchNormalization()(block)
block = keras.layers.Activation("relu")(block)
block = keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding="same")(block)

# #inio Squeeze and Excitation 3
# sq = keras.layers.GlobalAveragePooling2D()(block)
# sq = keras.layers.Reshape((1,1,64))(sq)
# sq = keras.layers.Dense(units=64,activation="sigmoid")(sq)
# block = keras.layers.multiply([block,sq])
# #fine Squeeze and Excitation 3

net = keras.layers.add([x,block])
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)
net = keras.layers.MaxPooling2D(pool_size=(2, 2),name="block_3")(net)

#net = keras.layers.GlobalAveragePooling2D()(net)


net = keras.layers.Flatten()(net)

shared1 = keras.layers.Dense(units=10)
shared2 = keras.layers.Dense(units=nclasses,activation="softmax")


d_net = []
for _ in range(N_SAMPLES):
    n = keras.layers.Dropout(0.5)(net)
    n = shared1(n)
    n = shared2(n)
    d_net.append(n)
net = keras.layers.Average()(d_net)


model = keras.models.Model(inputs=images,outputs=net)



def custom_loss_funct(y_true, y_pred):
    loss = keras.losses.categorical_crossentropy(y_true, d_net[0])
    for i in range(1,len(d_net)):
        loss += keras.losses.categorical_crossentropy(y_true, d_net[i])
    loss = loss/len(d_net)
    return(loss)


model.summary()


# In[ ]:


from IPython.display import SVG
import IPython
from keras.utils import model_to_dot

print(model.summary())

keras.utils.plot_model(model, to_file='test_keras_plot_model.png', show_shapes=True)
IPython.display.Image('test_keras_plot_model.png')


# In[ ]:


model.compile(loss=custom_loss_funct,
              optimizer='adadelta',
              metrics=['accuracy'])
checkpointer = keras.callbacks.ModelCheckpoint(filepath = 'cnn_from_scratch_fruits.hdf5', verbose = 1, save_best_only = True)
earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)


# In[ ]:


history=model.fit(x_train, y_train, batch_size=64, epochs=15,validation_data=(x_val, y_val), callbacks = [checkpointer,earlystopper], shuffle=True)


# In[ ]:


model.load_weights('cnn_from_scratch_fruits.hdf5')


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[ ]:


# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# **Visualization** Internal rappresentation

# In[ ]:


test_image = keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(train_dir+"/Apple Braeburn/0_100.jpg",target_size=(target_size, target_size)))
test_image = test_image/255

plt.imshow(test_image)


# In[ ]:


fig = plt.figure(figsize=(8, 8))
for img in range(3):
    ax = fig.add_subplot(1, 3, img+1)
    ax = plt.imshow(test_image[:, :, img],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    fig.subplots_adjust(wspace=0.05, hspace=0.05)


test_image = np.expand_dims(test_image, axis=0)


# In[ ]:


hidden_rappresenter = keras.models.Model(inputs=model.input, outputs=model.get_layer('block_1').output)
result=hidden_rappresenter.predict(test_image)
result.shape

fig = plt.figure(figsize=(16, 16))
for img in range(16):
    ax = fig.add_subplot(4, 4, img+1)
    ax = plt.imshow(result[0, :, :, img], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    fig.subplots_adjust(wspace=0.05, hspace=0.05)


# In[ ]:


hidden_rappresenter = keras.models.Model(inputs=model.input, outputs=model.get_layer('block_2').output)
result=hidden_rappresenter.predict(test_image)
result.shape

fig = plt.figure(figsize=(16, 16))
for img in range(32):
    ax = fig.add_subplot(6, 6, img+1)
    ax = plt.imshow(result[0, :, :, img], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    fig.subplots_adjust(wspace=0.05, hspace=0.05)


# In[ ]:


hidden_rappresenter = keras.models.Model(inputs=model.input, outputs=model.get_layer('block_3').output)
result=hidden_rappresenter.predict(test_image)
result.shape

fig = plt.figure(figsize=(16, 16))
for img in range(64):
    ax = fig.add_subplot(8, 8, img+1)
    ax = plt.imshow(result[0, :, :, img], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    fig.subplots_adjust(wspace=0.05, hspace=0.05)


# In[ ]:


y_test_pred = model.predict(x_test)
accuracy_score(np.argmax(y_test_pred,axis=1), np.argmax(y_test,axis=1))


# In[ ]:




