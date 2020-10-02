#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ##Importing the libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Loading the Data

# In[ ]:


dataset = pd.read_csv('../input/hiring.csv')

dataset.head()


# In[ ]:


dataset


# [](http://) The first col. 'experience' having numbers written in text which needs to be converted into numeric values also frst two fields are NaN which can be considered as '0' and need to fill those fields

# In[ ]:


dataset.experience


# In[ ]:


dataset.experience = dataset.experience.fillna('zero')

dataset.experience


# In[ ]:


from  word2number import w2n

dataset.experience = dataset.experience.apply(w2n.word_to_num)


# In[ ]:


dataset.experience


# Now the col. 'test_score' is having one NaN value which needs to be filled with some values. Here it will be filled with median value.

# In[ ]:


dataset.columns


# In[ ]:


import math
median_test = math.floor(dataset['test_score(out of 10)'].median())


# In[ ]:


median_test


# In[ ]:


dataset['test_score(out of 10)'] = dataset['test_score(out of 10)'].fillna(median_test)


# In[ ]:


dataset


# Now the data preprocessing has been done, need to assign the X and y features

# In[ ]:


X = dataset.iloc[: , :3].values

y = dataset.iloc[:, -1].values


# In[ ]:


X


# In[ ]:


y


# In[ ]:


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X, y)


# In[ ]:


regressor.coef_


# In[ ]:


regressor.intercept_


# In[ ]:


regressor.score(X, y)


# In[ ]:


regressor.predict([[5,6,7]])


# In[ ]:


regressor.predict([[2,10,10]])


# In[ ]:


regressor.predict([[2,9,6]])


# In[ ]:


X,y


# In[ ]:




