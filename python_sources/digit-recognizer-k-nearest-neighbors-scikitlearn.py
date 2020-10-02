#!/usr/bin/env python
# coding: utf-8

# ## This is a solution for [digit-recognizer(MNIST)](http://https://www.kaggle.com/c/digit-recognizer) using scikitlearn's K Nearest Neighbors, it is very easy to implement and it gives a not bad score of **0.968** on the official dataset

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


# In[ ]:


#load data
df = pd.read_csv("../input/train.csv")


# In[ ]:


# Take out label column
labels = df['label']
df = df.drop('label', 1)


# In[ ]:


# Let's see how data looks like 
image = df.values[123]
plt.imshow(image.reshape(28, 28), cmap='Greys')
plt.show()


# In[ ]:


# Split data
x_train, x_test, y_train, y_test = train_test_split(df.values, labels, test_size = 0.3, random_state = 0)

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)


# In[ ]:


# Setup KNN model
clf = KNeighborsClassifier()


# In[ ]:


#Train model
clf.fit(x_train, y_train)


# In[ ]:


# Predict
predictions = clf.predict(x_test)


# In[ ]:


# evaluate predictions
print ("Overal accuracy:", accuracy_score(predictions, y_test))
print (classification_report(predictions, y_test))
print (confusion_matrix(predictions, y_test))

