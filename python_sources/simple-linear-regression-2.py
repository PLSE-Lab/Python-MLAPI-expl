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
import matplotlib.pyplot as plt
import math 
from sklearn import linear_model
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


first_data = pd.read_csv("../input/random-linear-regression/train.csv")
second_data =pd.read_csv("../input/random-linear-regression/test.csv")


# In[ ]:


train_set = first_data.dropna()
test_set = second_data.dropna()


# In[ ]:


print(train_set.head())
print(test_set.head())


# In[ ]:


X = train_set[['x']].as_matrix()
y= train_set[['y']].as_matrix()


# In[ ]:


Xtest  = test_set[['x']].as_matrix()
ytest = test_set[['y']].as_matrix()


# In[ ]:


plt.figure(figsize=([8,6]))
plt.title("let see the realation b/w x and y of traning set")
plt.scatter(X,y,s=5,c="black",marker="*")
plt.xlabel("traning_set_x")
plt.ylabel("traning_set_y")
plt.show()


# In[ ]:


regression = linear_model.LinearRegression()


# In[ ]:


regression.fit(X,y)


# In[ ]:


regression.score(X,y)


# In[ ]:


math.sqrt(regression.score(X,y))


# In[ ]:


predic = regression.predict(X)


# In[ ]:


plt.figure(figsize=([8,6]))
plt.title("scatter b/w predicted and actual values")
plt.scatter(predic,y,s=5,c="red")
plt.xlabel("actual")
plt.ylabel("predicted")
plt.show()


# In[ ]:


predict = regression.predict(Xtest)


# In[ ]:


plt.figure(figsize=([8,6]))
plt.title("scatter b/w predicted and actual values in test set")
plt.scatter(ytest,predict,s=5,c="cyan")
plt.xlabel("test values")
plt.ylabel("predicted values")
plt.show()


# In[ ]:




