#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train_data = pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')
train_data.head()


# In[ ]:


test_data = pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')
test_data.head()


# In[ ]:


test_data.info()


# In[ ]:


train_data.info()


# In[ ]:


train_data['Slope'].plot(kind='hist')


# In[ ]:




test_data['Slope'].plot(kind='hist')
# In[ ]:


train_data['Elevation'].plot(kind='hist')


# In[ ]:


test_data['Elevation'].plot(kind='hist')


# In[ ]:


train_data['Cover_Type'].value_counts()#from results it is visible that nothing is over sampled or under sampled


# In[ ]:


from sklearn.model_selection import train_test_split 


# In[ ]:


X=train_data.drop(labels=['Id','Cover_Type'],axis=1)
y=train_data['Cover_Type']


# In[ ]:


X_train, X_val, y_train,y_val = train_test_split(X,y,random_state=40)


# In[ ]:


print(X_train.shape,y_train.shape)
print(X_val.shape,y_val.shape)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc=RandomForestClassifier(n_estimators=70)


# In[ ]:


rfc.fit(X_train,y_train)


# In[ ]:


rfc.score(X_val,y_val)


# In[ ]:


predict=rfc.predict(test_data.drop(labels=['Id'],axis=1))


# In[ ]:


submission = pd.DataFrame(data=predict,columns=['Cover_Type'])
submission.head()


# In[ ]:


submission['Id'] = test_data['Id']
submission.set_index('Id',inplace=True)


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('Submission.csv')


# In[ ]:




