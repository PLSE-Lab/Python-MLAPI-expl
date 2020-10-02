#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 


# In[ ]:


wine=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')


# In[ ]:


#datasets
wine


# In[ ]:


wine['quality'].value_counts()


# In[ ]:



sns.distplot(wine['quality'])


# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x='quality' ,y='fixed acidity', data=wine)


# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x='quality' ,y='citric acid', data=wine)


# In[ ]:


plt.figure(figsize=(10,6))

   sns.barplot(x='quality' ,y='alcohol', data=wine)


# In[ ]:


#quality as bad or good , bad<6.5 ,good>6.5 t o 8

bound = (2, 6.5, 8)
catg = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bound, labels = catg)


# In[ ]:


wine['quality']


# In[ ]:


wine['quality'].value_counts()


# In[ ]:


#data preprocessing 
from sklearn.preprocessing import LabelEncoder


# In[ ]:


encode=LabelEncoder()


# In[ ]:


wine['quality']=encode.fit_transform(wine['quality'])


# In[ ]:


wine['quality']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


#divide into train or test
X=wine.drop(['quality'], axis=1)
y=wine['quality']
X_train ,X_test, y_train, y_test = train_test_split(X ,y, test_size=0.30, random_state=101)


# In[ ]:


#sklearn model for datasests
from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:




