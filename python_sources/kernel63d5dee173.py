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


#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


dataset = pd.read_csv('/kaggle/input/iris/Iris.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values


# In[ ]:


dataset.head()


# In[ ]:


dataset.isnull().sum()


# In[ ]:


#spliting data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)


# In[ ]:


#since Y is categorical
#labeling first
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
Label = LabelEncoder()
Y_train = Label.fit_transform(Y_train)
Y_test = Label.transform(Y_test)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,Y_train)


# In[ ]:


y_predict = nb.predict(X_test)


# In[ ]:


y_predict


# In[ ]:


Y_test


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test,y_predict)
cm


# In[ ]:


accuracy_score(Y_test,y_predict)


# In[ ]:




