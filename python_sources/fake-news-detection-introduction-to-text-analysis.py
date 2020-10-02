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
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data=pd.read_csv('/kaggle/input/textdb3/fake_or_real_news.csv')


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


sns.countplot(data['label'])   #Unbiased Data


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(data['text'],data['label'],test_size=0.3,random_state=5)


# In[ ]:


print('Shape of X-Train : ', X_train.shape, '\n', 
     'Shape of X-Test : ', X_test.shape, '\n', 
     'Shape of Y-Train : ', y_train.shape, '\n', 
     'Shape of Y-Test : ', y_test.shape)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


tfid=TfidfVectorizer(stop_words='english',max_df=0.7) #Max DOCUMENT FREQUENCY df


# In[ ]:


transformed_Xtrain=tfid.fit_transform(X_train)
transformed_Xtest=tfid.transform(X_test)


# In[ ]:


from sklearn.linear_model import PassiveAggressiveClassifier


# In[ ]:


passive=PassiveAggressiveClassifier(max_iter=100)


# In[ ]:


passive.fit(transformed_Xtrain,y_train)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test,passive.predict(transformed_Xtest))

