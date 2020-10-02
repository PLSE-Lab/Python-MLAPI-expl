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


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# In[ ]:


train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")
test= pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


train


# In[ ]:


Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)


# In[ ]:


Y_test = test["label"]
X_test = test.drop(labels = ["label"],axis= 1)


# In[ ]:


#plotting some of the samples  
plt.subplot(2,2,1)
img1 = X_train.iloc[0].to_numpy().reshape((28,28))
plt.imshow(img1,cmap='gray')
plt.subplot(2,2,2)
img2 = X_train.iloc[10].to_numpy().reshape((28,28))
plt.imshow(img2,cmap='gray')
plt.subplot(2,2,3)
img3 = X_train.iloc[98].to_numpy().reshape((28,28))
plt.imshow(img3,cmap='gray')
plt.subplot(2,2,4)
img4 = X_train.iloc[25].to_numpy().reshape((28,28))
plt.imshow(img4,cmap='gray')
plt.show()


# In[ ]:


import keras
from keras.utils import np_utils
num_classes=len(np.unique(Y_train))
Y_train=keras.utils.to_categorical(Y_train,num_classes)
Y_test=keras.utils.to_categorical(Y_test,num_classes)

(X_train,X_valid)=X_train[5000:],X_train[:5000]

(Y_train,Y_valid)=Y_train[5000:],Y_train[:5000]

print('x_train shape:',X_train.shape)

print(X_train.shape[0],'train samples')
print(X_test.shape[0],'test samples')
print(X_valid.shape[0],'validation samples')


# In[ ]:


#Reshape 
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
X_valid = X_valid.values.reshape(-1,28,28,1)
print("X_train shape : ",X_train.shape)
print("Test shape : ",X_test.shape)
print("Test shape : ",X_valid.shape)


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)
X_train.shape


# In[ ]:





# In[ ]:





# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD, Adam

model = Sequential()

model.add(Convolution2D(32, (3, 3), input_shape=(28,28,1),activation='relu',padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Convolution2D(32, (3, 3), activation='relu',padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))   
model.add(Dropout(0.2))

model.add(Convolution2D(2*32, (3, 3), activation='relu',padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Convolution2D(2*32, (3, 3), activation='relu',padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))   
model.add(Dropout(0.2))

model.add(Convolution2D(4*32, (3, 3), activation='relu',padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Convolution2D(4*32, (3, 3), activation='relu',padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))   
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='elu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))
model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[ ]:


#from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
checkpointer=ModelCheckpoint(filepath='Convolutional.hdf5',verbose=1,save_best_only=True)
hist=model.fit_generator(datagen.flow(X_train,Y_train,batch_size=32),epochs=25,validation_data=(X_valid,Y_valid),callbacks=[checkpointer],verbose=1)


# In[ ]:


model.load_weights('Convolutional.hdf5')                  #using load_weights function 
score=model.evaluate(X_test,Y_test,verbose=1)               #evaluates the model
accuracy=100*score[1]                                       
print('Test accuracy is %.4f%%' % accuracy)                 #returns the accuracy of the model 


# In[ ]:


model.load_weights('Convolutional.hdf5')                  #using load_weights function 
score=model.evaluate(X_valid,Y_valid,verbose=1)               #evaluates the model
accuracy=100*score[1]                                       
print('Test accuracy is %.4f%%' % accuracy)                 #returns the accuracy of the model 


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[ ]:




