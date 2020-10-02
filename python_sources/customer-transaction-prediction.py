#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('../input/train.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.shape


# In[6]:


data.describe()


# In[7]:


data['target'].value_counts().plot(kind='bar')


# In[8]:


data.isnull().sum()


# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error


# In[10]:


y = data['target']
X = data.drop(['ID_code','target'],axis=1)


# In[11]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[13]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


# In[14]:


log_model = LogisticRegression()


# In[15]:


log_model.fit(X_train,y_train)


# In[17]:


y_pred = log_model.predict(X_test)
print('Accuracy: ',log_model.score(X_test,y_test))


# In[24]:


test_data = pd.read_csv('../input/test.csv')
X_test_test = test_data.drop(['ID_code'],axis =1)

y_pred_test = log_model.predict(X_test_test)

output = pd.DataFrame({'ID_code': test_data.ID_code,'target': y_pred_test})
output.to_csv('submission2.csv', index=False)
output.head()


# In[ ]:




