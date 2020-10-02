#!/usr/bin/env python
# coding: utf-8

# ****Loaded the 4 category files from quick draw as below. Since flowers not available i have choosen the Animals and Fruits****
# 
# ****The problem is an image classification hence i have used the quick draw numpy bitmap full files****

# In[ ]:


import os
os.listdir("../input")


# ** Load_Data function will help to load the npy file and convert to numpy arrays **

# In[ ]:


import numpy as np
import os
import pickle

files = os.listdir("../input")
x = []
x_load = []
y = []
y_load = []

def load_data(range_param):
    count = 0
    for file in files:
        file = "../input/" + file
        x = np.load(file)
        x = x.astype('float32') / 255.
        x = x[0:range_param, :]
        x_load.append(x)
        y = [count for _ in range(range_param)]
        count += 1
        y = np.array(y).astype('float32')
        y = y.reshape(y.shape[0], 1)
        y_load.append(y)

    return x_load, y_load


# In[ ]:


features, labels = load_data(100)
features = np.array(features).astype('float32')
labels = np.array(labels).astype('float32')
features=features.reshape(features.shape[0]*features.shape[1],features.shape[2])
labels=labels.reshape(labels.shape[0]*labels.shape[1],labels.shape[2])


# **This function is used to convert to one hot representation **

# In[ ]:


def prepress_labels(labels):
    labels = np_utils.to_categorical(labels)
    return labels


# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Dense,Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.utils import np_utils, print_summary
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pickle
from keras.callbacks import TensorBoard


# ** Splitted the data train and test with ration of 9:1 **

# In[ ]:


features, labels = shuffle(features, labels)
labels=prepress_labels(labels)
train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                        test_size=0.1)
train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)


# ** Created a CNN model as below with 2 layers of convolutional and pooling layers with fully connected layer**

# In[ ]:


num_of_classes = 4
image_x = 28
image_y = 28
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(image_x,image_y,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(num_of_classes, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
filepath = "QuickDraw.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# ** Binary Crossentrophy  gives a better result compare to other loss functions**

# In[ ]:


print_summary(model)
model.fit(train_x, train_y, validation_split=0.1, epochs=100, batch_size=16)
model.save('QuickDraw.h5')


# In[ ]:


test_pred = model.predict_classes(test_x)


# ** Test Set accuracy score **

# In[ ]:


model.evaluate(test_x, test_y)


# ** Converting One Hot to classes **

# In[ ]:


test_act = []
for i in range(test_y.shape[0]):
    test_act.append(np.argmax(test_y[i]))
    


# **Confusion Matrix will give a idea about predicting inter category**

# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(test_act, test_pred)


# ** Accuracy between predicted and actual**

# In[ ]:


accuracy_score(test_act, test_pred)


# In[ ]:


from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(test_act, test_pred, average='weighted')

