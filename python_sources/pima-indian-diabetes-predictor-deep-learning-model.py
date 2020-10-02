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


# PIMA Indian Diabetes dataset is very popular for pbuilding and training a deep neural network that can predict diabetes for women who are aged greater than 21. Lets build the model.

# Now lets add some necessary libraries!

# In[ ]:


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# Now lets read dataset for diabetes prediction.

# In[ ]:


import pandas as pd
diabetes = pd.read_csv("../input/diabetes.csv")


# In[ ]:


As the dataset has total 8 features, so we need to preprocess the data and visualize for feature show.


# In[ ]:


_=df.hist(figsize=(12, 10))
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

sc=StandardScaler()
X=sc.fit_transform(df.drop('Outcome', axis=1))
y=df['Outcome'].values
y_cat=to_categorical(y)
print(X.shape)


# Now lets build the neural model.

# In[ ]:



from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test=train_test_split(X,y_cat, random_state=22, test_size=0.2)

model= tf.keras.Sequential()
model.add(keras.layers.Dense(64, input_shape=(8,), activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))

model.compile(keras.optimizers.Adam(lr=0.05),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())


# Lets train the model.

# In[ ]:


model.fit(X_train, y_train, epochs=20, verbose=2, validation_split=0.1)


# Now lets get the prediction from the model based on test set.

# In[ ]:


y_pred=model.predict(X_test)


# **Now getting the results- accuracy, confusion matrix**

# In[ ]:


y_test_class=np.argmax(y_test, axis=1)
y_pred_class=np.argmax(y_pred, axis=1)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

accuracy_score(y_test_class, y_pred_class)
print(classification_report(y_test_class, y_pred_class))
print(confusion_matrix(y_test_class, y_pred_class))


# 

# In[ ]:





# In[ ]:





# In[ ]:





# 

# 
