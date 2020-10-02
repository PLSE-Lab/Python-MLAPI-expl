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


# #### Get the current working directory

# In[ ]:


import os
PATH = "/kaggle/input/face-classification/Data"
Test_PATH = "/kaggle/input/face-classification/Data/Test"


# #### Define the data path

# In[ ]:


DATA_PATH = os.path.join(PATH, 'Train')
data_dir_list = os.listdir(DATA_PATH)


# #### Get the list of folders inside data path

# In[ ]:


print(data_dir_list)


# #### Required variables declaration and initialization

# In[ ]:


img_rows=224
img_cols=224
num_channel=3

num_epoch=100
batch_size=32

img_data_list=[]
classes_names_list=[]


# In[ ]:


target_column = []


# #### Read the images and store them in the list

# In[ ]:


import cv2

for dataset in data_dir_list:
    classes_names_list.append(dataset) 
    print ('Loading images from {} folder\n'.format(dataset)) 
    img_list=os.listdir(DATA_PATH+'/'+ dataset)
    target_column.append(dataset)
    for img in img_list:
        input_img=cv2.imread(DATA_PATH + '/'+ dataset + '/'+ img )
        input_img_resize=cv2.resize(input_img,(img_rows, img_cols))
        img_data_list.append(input_img_resize)
        


# #### Get the number of classes

# In[ ]:


input_img.shape 


# In[ ]:


num_classes = len(classes_names_list)
print(num_classes)


# ####  Image preprocessiong

# In[ ]:


import numpy as np

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255


# In[ ]:


print (img_data.shape)

#img_data = img_data.reshape(img_data.shape[0], img_data.shape[1], img_data.shape[2], num_channel)

#print (img_data.shape)


# In[ ]:


num_of_samples = img_data.shape[0]
input_shape = img_data[0].shape


# In[ ]:


from sklearn.preprocessing import LabelEncoder
Labelencoder = LabelEncoder()
classes = Labelencoder.fit_transform(target_column)
classes = np.ones((num_of_samples,), dtype='int64')


# Convert class labels to numberic using on-hot encoding
# and its not a classification problem.

# In[ ]:


classes


# In[ ]:


from keras.utils import to_categorical
classes = to_categorical(classes, num_classes)


# #### Shuffle the dataset

# In[ ]:


from sklearn.utils import shuffle

X, Y = shuffle(img_data, classes, random_state=2)


# #### Split the dataset

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[ ]:


Y


# ####  Defining the model

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


# In[ ]:


model = Sequential()

model.add(Conv2D(16,(3,3),activation = "relu", input_shape=input_shape))
model.add(Conv2D(16,(3,3),activation = "relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# ####  Compile the model

# #### Model Summary

# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
model.summary()


# In[ ]:


X_train.shape


# In[ ]:


y_train


# In[ ]:


hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=20, verbose=1, validation_data=(X_test, y_test))


# #### Evaluating the model

# In[ ]:


score = model.evaluate(X_test, y_test, batch_size=batch_size)

print('Test Loss:', score[0])
print('Test Accuracy:', score[1])


# In[ ]:


test_image = X_test[0:1]
print (test_image.shape)


# In[ ]:


print(model.predict(test_image))
print(model.predict_classes(test_image))
print(y_test[0:1])


# #### Predict and compute the confusion matrix

# In[ ]:


from sklearn.metrics import confusion_matrix

Y_pred = model.predict(X_test)
print(Y_pred)


# In[ ]:


y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)


# In[ ]:


print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))

