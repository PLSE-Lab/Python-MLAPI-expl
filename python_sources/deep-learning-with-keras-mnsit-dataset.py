#!/usr/bin/env python
# coding: utf-8

# # **Importing all the important libraries**
# 
# Starting my Kernel for Digit Recognizer.
# Before this, I have been working on the same problem with 20by20 Pixel MNSIT images in MATLAB.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import confusion_matrix

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # **Importing Keras**

# In[ ]:


from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.losses import categorical_crossentropy


# # **Load the Data**
# 
# Traning Data is saved in X and test data is X_test. These are saved as Dataframe object.
# 
# Most of the pixels are empty. Having values in only few pixels and logically it should be this way. 
# 
# Removing output 'label' from traning data and saving it as target variable.
# 
# The countplot tells us that training data is balanced or has almost equal number of samples for all the classes.
# 

# In[ ]:


X = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
X_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
# Drop 'label' column
y = X.label
X.drop(['label'], axis=1, inplace=True)

g = sns.countplot(y)
y.value_counts()

print(X.shape)


# Spliting data into train and cross-valdiaton.

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      train_size=0.8, test_size=0.2,
                                                      random_state=1)


# Earlier operations were done on Dataframe, we are converting it to array for further processes.

# In[ ]:


X_train = np.array(X_train)
X_val = np.array(X_val)
y_train = np.array(y_train)
y_val = np.array(y_val)


# # **Visualization of Data**
# 
# This is to show what exactly do these numbers in 784 columns represent. In the second image assigned label (corresponding 'y') is in yellow box on top of the image.

# In[ ]:


fig = plt.figure(figsize= [3,3])
plt.imshow(X_train[3].reshape(28,28), cmap='gray')
ax.set_title(y_train[3])


# In[ ]:


import random
indices = range(len(y_train))
box = dict(facecolor='yellow', pad=5, alpha=1)

fig, ax = plt.subplots(10, 10, squeeze=True, figsize=(24,12))

for n in range(10):
    for m in range(10):
        d=random.choice(indices)
        ax[n][m].imshow(X_train[d].reshape(28,28), cmap='gray')
        ax[n][m].set_title(y_train[d],y=0.9,bbox=box)


# Reshaping the data for CNN

# In[ ]:


X_train = X_train.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)
print(X_train.shape, y_train.shape)


# The pixel fill strength is a vlaue between 0-255. Normalization gives better esults.

# In[ ]:


X_train = X_train.astype("float32")/255.
X_val = X_val.astype("float32")/255.


# One-Hot encoding the target variable

# In[ ]:


y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

print(y_train[0])


# # **Training the model**
# 
# Model is sequential. I tried three layers of Conv2D. In second and third layer stride length is set to 2 and after first one there is 0.2 Dropout. Ih had added MaxPool2D layer but chose against it after completing the Deep Learning Course. It suggested the 'Stride Length' parameter in Conv2D covers this aspect more or less with similar effect. Before the final Dense Layer two more dense layers are added each followed by Dropout. Number of classes that we need finally is 10 (numbers 0-9) so last Dense layers has 10 nodes. 
# 
# Another point to note here is that as we are using Dropout, validation accuracy here may be lower that we can achieve with given layers. Because we are not overfitting the data here, it should give as better result on private dataset compared to results with model had we not used dropout.
# Moreover, I'm experimenting with different configuration of layers and parameters so output keeps changing based on that.

# In[ ]:


model = Sequential()
model.add(Conv2D(20, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1)))
# model.add(BatchNormalization())
model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', strides = 2))
model.add(Dropout(0.2))
model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])


# In[ ]:


datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 10)


# Learning Rate Scheduler lets you keep a high learning rate close to a crest in gradient descent and small when close to bottom of a 'valley'. This way code is optimized to take larger steps when it can. Model summary will list out all the layers we have in the model.

# In[ ]:


annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

model.summary()


# In[ ]:


hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size=16),
                           steps_per_epoch=500,
                           epochs=30, #Increase this when not commiting the Kernel or testing few changes
                           verbose=2,  #1 for ETA, 0 for silent
                           validation_data=(X_val, y_val),
                           callbacks=[annealer])


# In[ ]:


final_loss, final_acc = model.evaluate(X_val, y_val, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))


# In[ ]:


plt.plot(hist.history['loss'], color='b')
plt.plot(hist.history['val_loss'], color='r')
plt.show()
plt.plot(hist.history['acc'], color='b')
plt.plot(hist.history['val_acc'], color='r')
plt.show()


# Confusion Matrix helps you identify which type of data maybe causing the higher losses and lower accuracy. Y-axis is True Lable and X-axis is Predicted Label. In some cases I get  16 '9's being read as '5's. This makes sense as sometimes a nine can look like a five if upper segment and middle segment is not connected well. Such mistkaes in recognizing digits can be made by humans as well. iwht more epochs such false positives are reduced.

# In[ ]:


y_hat = model.predict(X_val)
y_pred = np.argmax(y_hat, axis=1)
y_true = np.argmax(y_val, axis=1)
confusion_matrix(y_true, y_pred)


# **Submit**

# In[ ]:


X_test = np.array(X_test)
X_test = X_test.reshape(-1, 28, 28, 1)/255.


# In[ ]:


y_hat = model.predict(X_test, batch_size=64)


# In[ ]:


y_pred = np.argmax(y_hat,axis=1)


# In[ ]:


output_file = "submission.csv"
with open(output_file, 'w') as f :
    f.write('ImageId,Label\n')
    for i in range(len(y_pred)) :
        f.write("".join([str(i+1),',',str(y_pred[i]),'\n']))

