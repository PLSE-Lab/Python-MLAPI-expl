#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('writefile', 'MiniVggnet.py', 'from keras.models import Sequential\nfrom keras.layers.normalization import BatchNormalization\nfrom keras.layers.convolutional import Conv2D\nfrom keras.layers.convolutional import MaxPooling2D\nfrom keras.layers.core import Activation\nfrom keras.layers.core import Flatten\nfrom keras.layers.core import Dropout\nfrom keras.layers.core import Dense\nfrom keras import backend as K\n\n\nclass MiniVGGNet:\n    @staticmethod\n    def build(width, height, depth, classes):\n        # initialize the model along with the input shape to be\n        # "channels last" and the channels dimension itself\n        model = Sequential()\n        inputShape = (height, width, depth)\n        chanDim = -1\n        if K.image_data_format() == "channels_first":\n            inputShape = (depth, height, width)\n            chanDim = 1\n\n        model.add(Conv2D(32, (3, 3), padding="same",\n                         input_shape=inputShape))\n        model.add(Activation("relu"))\n        model.add(BatchNormalization(axis=chanDim))\n        model.add(Conv2D(32, (3, 3), padding="same"))\n        model.add(Activation("relu"))\n        model.add(BatchNormalization(axis=chanDim))\n        model.add(MaxPooling2D(pool_size=(2, 2)))\n        model.add(Dropout(0.25))\n\n        model.add(Conv2D(64, (3, 3), padding="same"))\n        model.add(Activation("relu"))\n        model.add(BatchNormalization(axis=chanDim))\n        model.add(Conv2D(64, (3, 3), padding="same"))\n        model.add(Activation("relu"))\n        model.add(BatchNormalization(axis=chanDim))\n        model.add(MaxPooling2D(pool_size=(2, 2)))\n        model.add(Dropout(0.25))\n\n        model.add(Flatten())\n        model.add(Dense(512))\n        model.add(Activation("relu"))\n        model.add(BatchNormalization())\n        model.add(Dropout(0.5))\n\n        model.add(Dense(classes))\n        model.add(Activation("softmax"))\n        # return the constructed network architecture\n        return model')


# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.optimizers import SGD
import MiniVggnet


# In[ ]:


dataset = pd.read_csv("../input/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data.csv").astype('float32')
dataset.rename(columns={'0':'label'}, inplace=True)

# Splite data the X - Our data , and y - the prdict label
X = dataset.drop('label',axis = 1)
y = dataset['label']


# In[ ]:


from sklearn.utils import shuffle

X_shuffle = shuffle(X)

plt.figure(figsize = (12,10))
row, colums = 4, 4
for i in range(16):  
    plt.subplot(colums, row, i+1)
    plt.imshow(X_shuffle.iloc[i].values.reshape(28,28),interpolation='nearest', cmap='Greys')
plt.show()


# In[ ]:


(trainX,testX,trainY,testY) = train_test_split(X/255.0,y.astype("int"),test_size=0.25,random_state=42)
standard_scaler = MinMaxScaler()
standard_scaler.fit(trainX)
trainX = standard_scaler.transform(trainX)
testX = standard_scaler.transform(testX)


# In[ ]:



if K.image_data_format() == 'channels_first':
  trainX = trainX.reshape(trainX.shape[0], 1, 28,28)
  testX = testX.reshape(trainX.shape[0], 1, 28, 28)
  input_shape = (1, 28, 28)
else:
  trainX = trainX.reshape(trainX.shape[0],28, 28, 1)
  testX = testX.reshape(testX.shape[0],28, 28, 1)
  input_shape = (28, 28, 1)


# In[ ]:


trainY = np_utils.to_categorical(trainY)
testY = np_utils.to_categorical(testY)


# In[ ]:


aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, fill_mode="nearest")


# In[ ]:


print("[INFO] compiling model....")

opt = SGD(lr=0.001)
model=MiniVggnet.MiniVGGNet.build(width=28,height=28,depth=1,classes=26)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])


# In[ ]:


print("[INFO] training network...")

H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
validation_data=(testX, testY), steps_per_epoch=len(trainX) // 32,
epochs=20, verbose=1)


# In[ ]:


from sklearn.metrics import classification_report
target_names=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
print("[INFO] evaluating network...")
predictions = model.predict(testX,batch_size=128)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),
                            target_names=target_names))


# In[ ]:


model.save("AlphabetrecognitionModel.hdf5")


# In[ ]:


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
plt.savefig('AlphabetrecogwithMiniVggnet.png')


# In[ ]:


from contextlib import redirect_stdout

with open('modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

