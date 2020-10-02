#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mlt
##!pip install pyarrow 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pyarrow as pa
import pyarrow.parquet as pq 


# In[ ]:


import pandas as pd
class_map = pd.read_csv("../input/bengaliai-cv19/class_map.csv")
sample_submission = pd.read_csv("../input/bengaliai-cv19/sample_submission.csv")
test = pd.read_csv("../input/bengaliai-cv19/test.csv")
train = pd.read_csv("../input/bengaliai-cv19/train.csv")


# In[ ]:


#class_map.head(2)


# class map has 185 . all unique component now moving on to the submission one. shape (186,3)

# In[ ]:


#sample_submission.head(2)


# shape (36,2). target , moving on to test

# In[ ]:


#test.head(2)


# test_id along with nothing to use. useless

# In[ ]:


#train.head(4)


# In[ ]:


y_train_grapheme_root=train["grapheme_root"]


# In[ ]:


y_train_grapheme_root.head(3)


# In[ ]:


y_train_vowel_diacritic=train["vowel_diacritic"]


# In[ ]:


y_train_consonant_diacritic=train["consonant_diacritic"]


# In[ ]:


del class_map
del sample_submission
del test
del train


# In[ ]:


test0 = pd.read_parquet("../input/bengaliai-cv19/test_image_data_0.parquet")


# In[ ]:


test0.head(5)


# In[ ]:


test0=test0.drop(["image_id"],axis=1)


# In[ ]:


test1 = pd.read_parquet("../input/bengaliai-cv19/test_image_data_1.parquet")
test1=test1.drop(["image_id"],axis=1)


# In[ ]:


test2 = pd.read_parquet("../input/bengaliai-cv19/test_image_data_2.parquet")
test2=test2.drop(["image_id"],axis=1)


# In[ ]:


test3 = pd.read_parquet("../input/bengaliai-cv19/test_image_data_3.parquet")
test3=test3.drop(["image_id"],axis=1)


# In[ ]:


x_test=pd.concat([test0,test1,test2,test3],ignore_index=True)


# In[ ]:


del test0
del test1
del test2
del test3


# In[ ]:


x_test.shape


# In[ ]:


x_test=x_test.values.reshape(-1,137,59,4)
g=plt.imshow(x_test[0][:,:,0])


# In[ ]:


train0=pd.read_parquet("../input/bengaliai-cv19/train_image_data_0.parquet")


# In[ ]:


train0.head(5)


# In[ ]:


train0=train0.drop(["image_id"],axis=1)


# In[ ]:


train1=pd.read_parquet("../input/bengaliai-cv19/train_image_data_1.parquet")
train1=train1.drop(["image_id"],axis=1)


# In[ ]:


train2=pd.read_parquet("../input/bengaliai-cv19/train_image_data_2.parquet")
train2=train2.drop(["image_id"],axis=1)


# In[ ]:


train3=pd.read_parquet("../input/bengaliai-cv19/train_image_data_3.parquet")
train3=train3.drop(["image_id"],axis=1)


# In[ ]:


x_train=pd.concat([train0,train1,train2,train3],ignore_index=True)


# In[ ]:


x_train.shape


# In[ ]:


x_train=x_train.values.reshape(-1,137,59,4)
g=plt.imshow(x_train[1000][:,:,0])


# In[ ]:


del train0
del train1
del train2
del train3


# **building a model.******

# In[ ]:


from tensorflow import keras
from tensorflow.keras import models,layers
model=models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(137,59 , 4)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(7, activation='softmax'))


# In[ ]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


#y_train_grapheme_root=y_train_grapheme_root.to_numpy()
x_train.shape


# In[ ]:


#x_train.head(2)


# In[ ]:


y_train_consonant_diacritic.shape


# In[ ]:


y_train_consonant_diacritic.head(2)


# In[ ]:


y_train_consonant_diacritic=y_train_consonant_diacritic.to_numpy()


# In[ ]:


model.fit(x_train, y_train_consonant_diacritic, epochs=1)


# In[ ]:


loss,acc=model.evaluate(x_train,y_train_consonant_diacritic)


# In[ ]:


pred_consonant_diacritic=model.predict(x_test)
pred_consonant_diacritic=np.argmax(pred_consonant_diacritic,axis=1)
dict={'consonant_diacritic':pred_consonant_diacritic}
submission=pd.DataFrame(dict)
submission.to_csv("pred_consonant_diacritic.csv")


# model for vowel diacritic ==11 complex model with more layers.

# In[ ]:


model1=models.Sequential()
model1.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(137,59 , 4)))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(layers.Flatten())
model1.add(layers.Dense(256,activation='relu'))
model1.add(layers.Dropout(0.1))
model1.add(layers.Dense(128,activation='relu'))
model1.add(layers.Dropout(0.1))
model1.add(layers.Dense(128,activation='relu'))
model1.add(layers.Dropout(0.2))
model1.add(layers.Dense(11, activation='softmax'))


# In[ ]:


model1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


y_train_vowel_diacritic=y_train_vowel_diacritic.to_numpy()


# In[ ]:


model1.fit(x_train, y_train_vowel_diacritic, epochs=1)


# In[ ]:


loss1,acc1=model.evaluate(x_train,y_train_vowel_diacritic)


# In[ ]:


pred_vowel_diacritic=model1.predict(x_test)
pred_vowel_diacritic=np.argmax(pred_vowel_diacritic,axis=1)
dict={'vowel_diacritic':pred_vowel_diacritic}
submission=pd.DataFrame(dict)
submission.to_csv("pred_vowel_diacritic.csv")


# model2 for grapheme root ===168. data augmentation needed.

# In[ ]:





# In[ ]:




