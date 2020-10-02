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

import os
import numpy as np
import cv2



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


model = keras.models.Sequential()

model.add(keras.layers.Conv2D(filters = 4, kernel_size = (3,3), activation='relu',input_shape=x_train.shape[1:],name="conv_1"))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.BatchNormalization(center=False, scale=False))

model.add(keras.layers.Conv2D(filters = 8, kernel_size = (3,3), activation='relu',name="conv_2"))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.BatchNormalization(center=False, scale=False))

model.add(keras.layers.Conv2D(filters = 16, kernel_size = (3,3), activation='relu',name="conv_3"))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.BatchNormalization(center=False, scale=False))


model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(8,activation="relu", name="dense"))
model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.Dense(nclasses,activation = 'softmax', name="output"))
model.summary()


# In[ ]:


from IPython.display import SVG
import IPython
from keras.utils import model_to_dot

print(model.summary())

keras.utils.plot_model(model, to_file='test_keras_plot_model.png', show_shapes=True)
IPython.display.Image('test_keras_plot_model.png')


# In[ ]:


model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
checkpointer = keras.callbacks.ModelCheckpoint(filepath = 'cnn_from_scratch_fruits.hdf5', verbose = 1, save_best_only = True)
earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='auto', baseline=None, restore_best_weights=False)


# In[ ]:


history=model.fit(x_train, y_train, batch_size=64, epochs=100,validation_data=(x_val, y_val), callbacks = [checkpointer,earlystopper], shuffle=True)


# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
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


# In[ ]:


y_test_pred = model.predict(x_test)
accuracy_score(np.argmax(y_test_pred,axis=1), np.argmax(y_test,axis=1))

