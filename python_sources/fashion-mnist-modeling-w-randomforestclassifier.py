#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


fashion_train = pd.read_csv('../input/fashion-mnist_train.csv')
fashion_test = pd.read_csv('../input/fashion-mnist_test.csv')


# In[ ]:


get_ipython().system('pip install idx2numpy')
import idx2numpy


# In[ ]:


file = '../input/t10k-images-idx3-ubyte'


# In[ ]:


array = idx2numpy.convert_from_file(file)


# In[ ]:


print(array[2])


# In[ ]:


plt.imshow(array[2], cmap = 'gray')
plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split
df = fashion_train
x = df.drop(['label'], axis = 1)
y = df.iloc[:, 0]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 25, max_depth = 12, random_state = 2)
model.fit(x_train, y_train)


# In[ ]:


pred = model.predict(x_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, pred))


# In[ ]:


print(classification_report(y_test, pred))

