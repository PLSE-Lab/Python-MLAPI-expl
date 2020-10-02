#!/usr/bin/env python
# coding: utf-8

# ## Cats and Dog Starter Code (adopting Sentdex approach)
# 1.  Useful for beginners
# 2.  Learn a crude alternative to creating your image training dataset without using keras flow_from_directory
# 

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import keras
import matplotlib.pyplot as plt
import cv2

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import Activation
from keras.layers import MaxPool2D
from keras.layers import Flatten

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


os.listdir('../input/training_set/training_set/cats')[:5]


# In[ ]:


#checking out the first image in the cats dataset
im = cv2.imread(('../input/training_set/training_set/cats/cat.2003.jpg'), cv2.IMREAD_GRAYSCALE)
plt.imshow(im)


# In[ ]:


#Looping each directory for cats and dogs individually
    #would be a useful exploration in integrating into dataset later

base_dir = '../input/training_set/training_set/'
categories = ['cats', 'dogs']
for category in categories:
    child_path = os.path.join(base_dir, category)
    for path in os.listdir(child_path):
        img_array = cv2.imread(os.path.join(child_path,path), cv2.IMREAD_GRAYSCALE)
        #GRAYSCALE slightly reduces array size of a colored image from say 
        #(_,255,255,3) to (_,255,255,1). Useful in cases where color do not really matter 
        plt.imshow(img_array)
        plt.show()
        break  #Let's see if our loop works perfectly before proceeding any further
    break


# In[ ]:


training_data = [] #creates empty list of training dataset
base_dir = '../input/training_set/training_set/'
categories = ['cats', 'dogs']
new_size = 100 #previously 255-max
for category in categories:
    child_path = os.path.join(base_dir, category)
    label = categories.index(category)
    for path in os.listdir(child_path):
        try: #some of the images in this dataset throw errors
            img_array = cv2.imread(os.path.join(child_path,path), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (new_size,new_size)) 
            training_data.append([new_array,label])
        except Exception as exc: 
            pass


# In[ ]:


import random
random.shuffle(training_data) #shuffles the training_data for better performance


# In[ ]:


for sample in training_data:
    sample[0] = sample[0]/255.0 #normalizes the dataset


# In[ ]:


X = []
y = []
for sample, features in training_data:
    X.append(sample)
    y.append(features)


# In[ ]:


X = np.array(X).reshape(-1,new_size,new_size,1)


# In[ ]:


y[:10]


# In[ ]:


y = np.array(y)
y = keras.preprocessing.utils.to_categorical(y) #one-hot encodes the labels for seamless integration into CNN


# In[ ]:


y[:5]


# In[ ]:


#preparing the test data inthe same fashion 
test_data = []
base_dir = '../input/test_set/test_set/'
categories = ['cats', 'dogs']
new_size = 100
for category in categories:
    child_path = os.path.join(base_dir, category)
    label = categories.index(category)
    for path in os.listdir(child_path):
        try:
            img_array = cv2.imread(os.path.join(child_path,path), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (new_size,new_size))
            test_data.append([new_array,label])
        except Exception as e:
            pass


# In[ ]:


random.shuffle(test_data)
for sampl in test_data:
    sampl[0] = sampl[0]/255.0
X_test = []
y_test = []
for sample, features in test_data:
    X_test.append(sample)
    y_test.append(features)


# In[ ]:


X_test = np.array(X_test).reshape(-1,new_size,new_size,1)
y_test = keras.preprocessing.utils.to_categorical(y_test)


# In[ ]:


y_test[0]


# In[ ]:


# creating CNN
model = Sequential()
model.add(Convolution2D(32,3,3, input_shape=X.shape[1:], activation='relu')) 
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', 
              metrics=['accuracy'],
              )


# In[ ]:


model.fit(X,y, verbose=1,
              epochs=10,
         validation_data=(X_test,y_test))


# In[ ]:


X.shape[1:]


# In[ ]:


# saving and testing our model
import pickle


# In[ ]:


pickle.dump(model, open("model.pickle", 'wb'))


# In[ ]:


pickle_model = pickle.load(open("model.pickle", 'rb'))


# In[ ]:


predictions = pickle_model.predict([X_test])


# In[ ]:


predictions = np.argmax(predictions, axis=1)


# In[ ]:


print(predictions[58])
plt.imshow(X_test[58].reshape(new_size,new_size))


# In[ ]:


print(predictions[95])
plt.imshow(X_test[95].reshape(new_size,new_size))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




