#!/usr/bin/env python
# coding: utf-8

# # Dogs v/s Cats Classification

# This is a classification problem which consists of 25000 images of dogs and cats in jpeg format. Our task is to correctly classify them as dogs and cats using convolutional neural networks.

# ## Importing Libraries

# We will be using keras framework for implementing our model

# In[ ]:


import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical
from keras.applications import MobileNetV2
from sklearn.metrics import classification_report
import cv2
from tqdm import tqdm


# In[ ]:


import os
image_ids = os.listdir('../input/train/train/')


# As the data consists of images named as "dog.1.jpg", so we will store the image information in the x_train list and the category into y_train list.

# In[ ]:


x_train = []
y_train = []
for i in tqdm(image_ids):
    category = i.split(".")[0]
    if category == "dog":
        y_train.append(1)
    else:
        y_train.append(0)
        
    img_arr = cv2.imread("../input/train/train/"+i, cv2.IMREAD_GRAYSCALE)
    #img_arr = cv2.imread("../input/train/train/"+i)
    img_arr = cv2.resize(img_arr, dsize=(128, 128))
    x_train.append(img_arr)


# In[ ]:


x_train = np.array(x_train)
x_train.shape


# In[ ]:


x_train = x_train/255
#x_train = x_train.reshape(-1, 128, 128, 3)
x_train = x_train.reshape(-1, 128, 128, 1)


# In[ ]:


x_train.shape


# In[ ]:


import pickle
f = open("x_train.pickle", "wb")
pickle.dump(x_train, f)
f.close()
f = open("y_train.pickle", "wb")
pickle.dump(y_train,f)
f.close()


# ## Defining our model

# In[ ]:


model = Sequential()
model.add(Conv2D(4,(3,3),strides=1, padding='valid', activation = 'relu', input_shape = x_train.shape[1:]))
model.add(MaxPooling2D(pool_size = (2,2), strides=2))

model.add(Conv2D(16,(3,3), activation = 'relu', strides=1, padding="valid"))
model.add(MaxPooling2D(pool_size = (2,2), strides=2))

model.add(Conv2D(32, (3,3), activation="relu", strides=1, padding="valid"))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(64, (3,3), activation="relu", strides=1, padding="valid"))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='sigmoid'))

model.add(Dense(1, activation='sigmoid'))


# In[ ]:


#base_model = MobileNetV2(input_shape=(128,128,3),
 #                        include_top=False, 
  #                       weights='imagenet')


# In[ ]:


#model = Sequential([base_model,
 #                   MaxPooling2D(),
  #                  Flatten(),
   #                 Dense(1, activation='sigmoid')])


# In[ ]:


model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


model.fit(x_train, y_train, epochs=1, batch_size=32, validation_split=0.2)


# In[ ]:


f = open("model.pickle", "wb")
pickle.dump(model, f)
f.close()


# In[ ]:


x_test = []
test_files = os.listdir("../input/test1/test1/")
for i in tqdm(test_files):    
    img_arr = cv2.imread("../input/test1/test1/"+i, cv2.IMREAD_GRAYSCALE)
    img_arr = cv2.resize(img_arr, dsize=(128, 128))
    x_test.append(img_arr)


# In[ ]:


x_test = np.array(x_test)/255
x_test = x_test.reshape(-1, 128, 128, 1)


# In[ ]:


x_test.shape


# In[ ]:


predictions = model.predict(x_test)


# In[ ]:


results = []
for i in predictions:
    if(i>0.5):
        results.append(1)
    else:
        results.append(0)


# In[ ]:


df = pd.DataFrame({"id":[i+1 for i in range(12500)], 
                   "lable" : [p for p in results]})


# In[ ]:


df.to_csv("submission.csv",index=False)


# In[ ]:




