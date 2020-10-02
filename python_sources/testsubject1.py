#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import seaborn as sns 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/iris-flower-dataset/IRIS.csv')


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.corr()


# In[ ]:


sns.heatmap(df.corr())


# In[ ]:


sns.pairplot(df,hue='species')


# In[ ]:


df.info()


# In[ ]:


print(df.isnull().sum())


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[ ]:


x = df.drop(['species'],axis=1)
y = df['species']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=np.random)


# In[ ]:


logmodel=LogisticRegression()


# In[ ]:


logmodel.fit(x_train,y_train)


# In[ ]:


predictions=logmodel.predict(x_test)


# In[ ]:


print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))


# In[ ]:




