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


# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


data = pd.read_table("/kaggle/input/fruit_data_with_colors.txt",sep='\t')


# In[ ]:


data.head(3)


# In[ ]:


x= data[['mass','width','height','color_score']]
y= data['fruit_label']


# In[ ]:


print(x.shape)
print(y.shape)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test= train_test_split(x,y)


# In[ ]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
knn.score(x_test,y_test)


# In[ ]:


data.groupby('fruit_name').agg('count')


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


mm = MinMaxScaler()
x = mm.fit_transform(x)


# In[ ]:


x


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
knn.score(x_test,y_test)

