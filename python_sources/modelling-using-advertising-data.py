#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[3]:


file = '../input/Advertising.csv'
data = pd.read_csv(file)


# In[4]:


data.head()


# In[5]:


data.drop(['Unnamed: 0'],axis = 1,inplace = True)


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


sns.heatmap(data.corr(),cmap = 'magma',lw = .7,linecolor = 'lime',alpha = 0.8,annot = True)


# In[ ]:


sns.distplot(data['sales'],hist = True)


# In[ ]:


sns.jointplot(x = 'sales',y = 'TV',data = data,kind = 'kde',color = 'red')


# In[ ]:


sns.jointplot(x = 'sales',y = 'newspaper',data = data,kind = 'kde',color = 'green',hist = True)


# In[ ]:


sns.jointplot(x = 'sales',y = 'radio',data = data,kind = 'kde',color = 'gold',hist = True)


# In[ ]:


sns.pairplot(data,height = 4)


# In[ ]:


X = data.drop(['sales'],axis = 1)
y = data['sales']


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:


logmodel =  LogisticRegression()


# In[ ]:


from sklearn import preprocessing
from sklearn import utils

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y_train)


# In[ ]:


lab_enc = preprocessing.LabelEncoder()
encoded2 = lab_enc.fit_transform(y_test)


# In[ ]:


logmodel.fit(X_train,encoded)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(encoded2,predictions))
print(confusion_matrix)

