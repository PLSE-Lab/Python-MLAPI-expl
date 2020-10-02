#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(1)
#Sklearn
from sklearn.utils import shuffle
#Keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras import regularizers
from random import randint


# **Hello every one this is my first Keras Convolutional Neural Network. I am still learning so please any suggestion or improvement in my codes would be appreciate**

# **Intel image dataset has provided six catagories of images to work on.**
# 1. Glaciers
# 2. Sea
# 3. Building
# 4. Forest
# 5. Street
# 6. Mountain
# <br><br>
# **So this is a multiclass detection.**

# Looping in directories of train and test data and converting images to tensor. Also resizing the image to 
# 150 x 150 x 3 so all the images have the same size.

# In[ ]:


images_train = []
labels_train = []
label = 0
for directories in os.listdir('../input/seg_train/seg_train/'):
    if directories == 'glacier':
        label = 0
    elif directories == 'sea':
        label = 1
    elif directories == 'buildings':
        label = 2
    elif directories == 'forest':
        label = 3
    elif directories == 'street':
        label = 4
    elif directories == 'mountain':
        label = 5
    for img in os.listdir('../input/seg_train/seg_train/' + directories):
        image = cv2.imread('../input/seg_train/seg_train/' + directories + '/' + img)
        image = cv2.resize(image,(150, 150))
        images_train.append(image)
        labels_train.append(label)
        Images_train, Labels_train = shuffle(images_train, labels_train)


# In[ ]:


images_test = []
labels_test = []
label = 0
for directories in os.listdir('../input/seg_test/seg_test/'):
    if directories == 'glacier':
        label = 0
    elif directories == 'sea':
        label = 1
    elif directories == 'buildings':
        label = 2
    elif directories == 'forest':
        label = 3
    elif directories == 'street':
        label = 4
    elif directories == 'mountain':
        label = 5
    for img in os.listdir('../input/seg_test/seg_test/' + directories):
        image = cv2.imread('../input/seg_test/seg_test/' + directories + '/' + img)
        image = cv2.resize(image,(150, 150))
        images_test.append(image)
        labels_test.append(label)
        images_test, labels_test = shuffle(images_test, labels_test)


# Converting the list of images to numpy arrary and normalizing it by dividing it with 255. We divide it with 255 because it is a color image and color range is 0 - 255 

# In[ ]:


train_x = np.array(images_train)/255
train_y = np.array(labels_train)
test_x = np.array(images_test)/255
test_y = np.array(labels_test)


# In[ ]:


print ('Train Y shape ', train_y.shape)
print ('Test Y shape ', test_y.shape)


# As you can see the shape of Y train and test is not right so reshaping them

# In[ ]:


train_y = train_y.reshape(train_y.shape[0], 1)
test_y = test_y.reshape(test_y.shape[0], 1)


# Changing Y train and test to **One Hot Encoding**

# In[ ]:


train_y = np_utils.to_categorical(train_y, 6)
test_y = np_utils.to_categorical(test_y, 6)


# In[ ]:


print("Shape of Images Train X:",train_x.shape)
print("Shape of Labels Train Y:",train_y.shape)
print()
print("Shape of Images Test X:",test_x.shape)
print("Shape of Labels Test Y:",test_y.shape)


# **Model Layout**

# In[ ]:


model = Sequential()
model.add(Conv2D(200, kernel_size=3, activation='relu', input_shape=(150, 150, 3)))
model.add(Conv2D(100, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(1,1)))
model.add(BatchNormalization(momentum=0.99, epsilon=0.01))
model.add(Dropout(0.2))

model.add(Conv2D(80, kernel_size=3, activation='relu'))
model.add(Conv2D(70, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(1,1)))
model.add(BatchNormalization(momentum=0.99, epsilon=0.01))
model.add(Dropout(0.2))

model.add(Conv2D(60, kernel_size=3, activation='relu'))
model.add(Conv2D(50, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(1,1)))
model.add(BatchNormalization(momentum=0.99, epsilon=0.01))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(45, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization(momentum=0.99, epsilon=0.01))

model.add(Dense(35, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization(momentum=0.99, epsilon=0.01))

model.add(Dense(15, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization(momentum=0.99, epsilon=0.01))

model.add(Dense(6, activation='softmax'))

learning_rate = 0.0001
opt = Adam(lr=learning_rate)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# Training Model

# In[ ]:


np.random.seed(1)

history = model.fit(train_x, train_y, batch_size=32, epochs=50)
results = model.evaluate(test_x, test_y)


# In[ ]:


plt.plot(np.squeeze(history.history["loss"]))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()
    
print("\n\nAccuracy on training set is {}".format(history.history["acc"][-1]))
print("\nAccuracy on test set is {}".format(results[1]))


# Freeing up memory

# In[ ]:


del train_x
del train_y
del test_x
del test_y
del images_train
del labels_train
del images_test
del labels_test


# Looping through "pred" folder and resizing images

# In[ ]:


images_pred = []
for img in os.listdir('../input/seg_pred/seg_pred/'):
    image = cv2.imread('../input/seg_pred/seg_pred/' + img)
    image = cv2.resize(image,(150, 150))
    images_pred.append(image)


# Normalizing and converting to numpy array

# In[ ]:


pred = np.array(images_pred)/255
print("The Predict Dataset Shape: ", pred.shape)


# Predicting the images with our trained model

# In[ ]:


predicting = model.predict_classes(pred, verbose=1)


# Labeling the outputs

# In[ ]:


predict_labels = []
for i in predicting:
    if i == 0:
        predict_labels.append('Glacier')
    elif i == 1:
        predict_labels.append('Sea')
    elif i == 2:
        predict_labels.append('Buildings')
    elif i == 3:
        predict_labels.append('Forest')
    elif i == 4:
        predict_labels.append('Street')
    elif i == 5:
        predict_labels.append('Mountain')


# Ploting predictions from the model. The Title of the images are the prediction of the model

# In[ ]:


fig = plt.figure(figsize=(20, 20))
for x in range(1, 21):
    ax = fig.add_subplot(5, 4, x)
    ax.set_title('Model Predicted = ' + predict_labels[x-1])
    ax.imshow(images_pred[x-1])

