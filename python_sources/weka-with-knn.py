#!/usr/bin/env python
# coding: utf-8

# In[92]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[93]:


data = pd.read_csv('../input/column_2C_weka.csv')


# In[ ]:


data.head()


# In[ ]:


data.dtypes


# In[ ]:


A = data[data['class'] == 'Abnormal']
N = data[data['class'] == 'Normal']


# In[ ]:


plt.scatter(A.lumbar_lordosis_angle, A['pelvic_radius'], color = 'red', label = 'abnormal')
plt.scatter(N.lumbar_lordosis_angle, N['pelvic_radius'], color = 'green', label = 'normal')
plt.xlabel('lumbar_lordosis_angle')
plt.ylabel('pelvic_radius')
plt.legend()
plt.show()


# In[94]:


data['class'] = [1 if i == 'Abnormal' else 0 for i in data['class']]


# In[95]:


x_data = data.drop(['class'], axis = 1)
y = data['class'].values


# In[96]:


x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[97]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)


# In[108]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 19) #n_neighbors = K
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)


# In[109]:


prediction


# In[110]:


knn.score(x_test, y_test)


# In[111]:


score_list = []
for i in range(1,25):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(x_train, y_train)
    score_list.append(knn2.score(x_test, y_test))
plt.plot(range(1,25), score_list)
plt.xlabel('k_values')
plt.ylabel('accuracy')
plt.show()


# In[ ]:




