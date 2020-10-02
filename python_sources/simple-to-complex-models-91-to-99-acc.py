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


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop, Adam
import pandas as pd
import matplotlib.pyplot as plt


# ## Reading and Preparing data
# Read the train and final tes data from file. Normalize the data, split the train data to train-test sets and visualize some samples. And check the balance of data set.

# In[ ]:


# Read data from file
data_train = np.loadtxt('../input/train.csv', skiprows=1, delimiter=',')
data_test = np.loadtxt('../input/test.csv', skiprows=1, delimiter=',')


# In[ ]:


# Normalize and split data set
data_test = data_test/255.0
X_train = data_train[:,1:]/255.0
y_train = data_train[:,0]
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
image_height = 28
image_width = 28


# In[ ]:


# Visualize some samples
nplot=10
fig, axes = plt.subplots(nplot, nplot, sharex=True, sharey=True, figsize=(9,9))
idex = np.arange(0,X_train.shape[0])
for i in range(nplot):
    for j in range(nplot):
        axes[i,j].imshow(X_train[np.random.choice(idex,1,replace=False)[0],:].reshape(image_height,image_width), cmap=plt.get_cmap('gray'))
        axes[i,j].set_xticklabels([])
        axes[i,j].set_yticklabels([])
        axes[i,j].axis('off')
plt.show()


# In[ ]:


# Check the balance of data set
x = np.arange(0,10)
numnum = []
for i in range(len(x)):
    numnum.append((y_train == x[i]).sum())
plt.bar(x, numnum, color='r')
plt.xticks(range(10))
plt.xlabel('Digit')
plt.ylabel('# of Samples')
plt.show()


# The data set is pretty balance so we don't need to use cost-sensitive trick like penalize

# ## Predictive models
# We will go from simple models like Logistic Regression to more complex Random Forest and to complex and advanced Convolutional Neural Network (CNN).
# ### Logistic Regression
# We need to tune the regularization prefactor parameter to avoid overfitting for logistic regression.

# In[ ]:


acc_lrg = []
for C in [0.00001, 0.001, 1.0, 100]:
    lrg = LogisticRegression(C=C)
    lrg.fit(X_train,y_train)
    acc_lrg.append([C, lrg.score(X_train, y_train), lrg.score(X_test, y_test)])
acc_lrg = np.array(acc_lrg)


# In[ ]:


plt.plot(np.log10(acc_lrg[:,0]), acc_lrg[:,1], marker='o', label='Training')
plt.plot(np.log10(acc_lrg[:,0]), acc_lrg[:,2], marker='s', label='Testing')
plt.xlabel(r'log$_{10}$(C)')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# For C = 1 we get the best tesing accuracy of ~ 0.91, which is very impresive already for this simple model. However, we need to go to complex and advanced CNN model which is known superior for image/visulizaton applications to get more than 0.99 accuracy.

# ## Random Forest

# In[ ]:


acc_rfc = []
for n_est in [10, 20, 40, 60, 80, 120]:
    rfc = RandomForestClassifier(n_estimators=n_est, bootstrap=False)
    rfc.fit(X_train, y_train)
    acc_rfc.append([n_est, rfc.score(X_train, y_train), rfc.score(X_test, y_test)])
acc_rfc = np.array(acc_rfc)


# In[ ]:


plt.plot(acc_rfc[:,0], acc_rfc[:,1], marker='o', label='Training')
plt.plot(acc_rfc[:,0], acc_rfc[:,2], marker='s', label='Testing')
plt.xlabel('# of Tree')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# We can get quite better accuracy of 0.965 with Random Forest model using 60 trees. The accuracy is already very high but as we will see latter that with comlex and advanced CNN model, we can get higher than 0.99!
# We already tried to increase number of tree to reduce overfit, but it looks clearly from plot that the model here is still having issues of overfit. More tuning other parameter such as min_samples_split or min_samples_leaf may help reduce more overfitting.

# ## Convolutional Neural Network (CNN)
# Preparing data for CNN model: converting categorial label of digit to vector representation.

# In[ ]:


n_class = 10
y_train_nn = keras.utils.to_categorical(y_train, n_class)
y_test_nn = keras.utils.to_categorical(y_test, n_class)
X_train_cnn = X_train.reshape(X_train.shape[0], image_height, image_width, 1)
X_test_cnn = X_test.reshape(X_test.shape[0], image_height, image_width, 1)


# In this CNN model, several Dropout and BatchNormalization layers will be used. Dropout layer will help to reduce overfit. BatchNormalization layers will help to stabilize the learing process and it also partially help reducing overfit as it introduces some noises to the learning process.

# In[ ]:


# Define model
cnn_model = Sequential()
cnn_model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(image_height, image_width, 1)))
cnn_model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Dropout(0.25))

cnn_model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
cnn_model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Flatten())

cnn_model.add(BatchNormalization())
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dropout(0.25))
cnn_model.add(Dense(n_class, activation='softmax'))

cnn_model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
cnn_model.summary()


# In[ ]:


# Training the model
cnn_model_his = cnn_model.fit(X_train_cnn, y_train_nn, epochs=20, validation_data=(X_test_cnn, y_test_nn))


# With 10 epoch we already can get accuracy of > 0.99 for test set.

# Submit my prediction

# In[ ]:


# Reshape input data
data_test_cnn = data_test.reshape(data_test.shape[0], image_height, image_width, 1)
answer = cnn_model.predict(data_test_cnn)
answer = np.argmax(answer, axis=1)
submission = pd.Series(answer, name='Label')
submission = pd.concat([pd.Series(range(1,28001), name='ImageId'), submission], axis=1)
submission.to_csv("my_submission.csv", index=False)

