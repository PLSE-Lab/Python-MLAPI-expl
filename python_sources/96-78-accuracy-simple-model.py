#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df


# In[ ]:


y=df.iloc[:,0]
y


# In[ ]:


y.value_counts()


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()


# In[ ]:


x=df.iloc[:,1:]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.00001,random_state=2)


# In[ ]:


model.fit(x_train,y_train)
model.score(x_train,y_train)


# In[ ]:


df_test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# In[ ]:


df_test


# In[ ]:


predicted=model.predict(df_test)
predicted


# In[ ]:


submit=pd.DataFrame({'ImageId':list(range(1,len(predicted)+1)),'Label':predicted})


# In[ ]:


submit.to_csv('mnist_submission.csv',index=False)

