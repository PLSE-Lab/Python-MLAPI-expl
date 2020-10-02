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


dataset=pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")


# In[ ]:


dataset.tail()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
dataset.iloc[:,1:2]=l.fit_transform(dataset.iloc[:,1:2])


# In[ ]:


dataset=dataset.iloc[:,:32]
dataset.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler
s=StandardScaler()
dataset.iloc[:,2:]=s.fit_transform(dataset.iloc[:,2:])


# In[ ]:


y=dataset.iloc[:,1:2]
x=dataset.iloc[:,2:]


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)


# In[ ]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)


# In[ ]:


prediction=model.predict(x_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,prediction)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction)


# In[ ]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,prediction)


# In[ ]:




