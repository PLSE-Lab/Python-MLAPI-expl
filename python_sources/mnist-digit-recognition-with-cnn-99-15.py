#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 

get_ipython().system('pip install tensorlayer')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# train = pd.read_csv('/kaggle/input/mnist-70000-original/MNIST_data.csv')
# target = pd.read_csv('/kaggle/input/mnist-70000-original/MNIST_target.csv')
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
Y = train['label']
X = train.drop('label', axis=1)
# Y=target
# X=train

# xx = np.array(train.drop("label", axis=1)).astype('float32')
# yy = np.array(train['label']).astype('float32')
# for i in range(9):
#     plt.subplot(3,3,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(xx[i].reshape(28, 28), cmap=plt.cm.binary)
#     plt.xlabel(yy[i])
# plt.show()


# Pre-processing

# In[ ]:


X = X/255.
test_X = test/255.


# In[ ]:


# Using Data augmentation to prevent overfitting
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=50,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = False, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

X = X.values.reshape(-1,28,28,1)
test_X = test_X.values.reshape(-1,28,28,1)
datagen.fit(X)


# In[ ]:


from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
Y = label_binarizer.fit_transform(Y)


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# X_train=X
# y_train=Y


# CNN

# In[ ]:


import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout,BatchNormalization, GlobalAveragePooling2D

def create_model():
    model = models.Sequential()
    model.add(Conv2D(100 , (5,5) , strides = 2 , padding = 'same', activation = 'relu' , input_shape = (28,28,1)))
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(BatchNormalization())
    model.add(Conv2D(75 , (5,5) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(BatchNormalization())
    model.add(Conv2D(50 , (5,5) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(units = 256 , activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(units = 10 , activation = 'softmax')) 
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model


# In[ ]:


from sklearn.metrics import mean_squared_error

# early = tf.keras.callbacks.EarlyStopping(patience=3)

model = create_model()
# history = model.fit(X,Y, epochs=20, validation_data=(x_test, y_test), verbose=1)
history = model.fit_generator(datagen.flow(X, Y, batch_size = 128), epochs=20, validation_data=(x_test, y_test), verbose=1)
pred_y = model.predict(x_test)
np.sqrt(mean_squared_error(y_test, pred_y))


# In[ ]:


# test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
# p = model.predict_classes(x_test)
# pp = y_test.argmax(axis=1)
# print((pp!=p).sum())
# j=0
# figure = plt.figure(figsize=(5, 5))
# for i in range(len(p)):
#     if(p[i] != pp[i]):
#         plt.subplot(5,5,j+1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.binary)
#         plt.xlabel("Real: %d, Predicted: %d" % (pp[i], p[i]))
#         j += 1
#     if(j==25):
#         break
# plt.tight_layout()
# plt.show()


# In[ ]:


# from sklearn.model_selection import KFold
 
# n_split=2

# best = None
# best_eval = None

# early = tf.keras.callbacks.EarlyStopping(patience=3)

# for train_index,test_index in KFold(n_split).split(X):
#     x_train,x_test=X[train_index],X[test_index]
#     y_train,y_test=Y[train_index],Y[test_index]

#     model=create_model()
#     model.fit(datagen.flow(X, Y, batch_size = 256), callback=[early], epochs=30)
#     evaluate=model.evaluate(x_test,y_test)
#     if not best or evaluate > best_eval:
#         best = model
#         best_eval = evaluate
#         print('Model evaluation ', evaluate)

# print('Best ', best_eval)


# In[ ]:


predictions = model.predict_classes(test_X)
# predictions = label_binarizer.inverse_transform(predictions)
submission = pd.DataFrame({'ImageId': range(1,len(predictions)+1), 'Label': predictions})
submission.to_csv('digit_cnn.csv', index=False)
submission.head()
# predictions

