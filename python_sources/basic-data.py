#!/usr/bin/env python
# coding: utf-8

# In[3]:


import xgboost
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import cross_validation, metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score


# In[4]:


train = pd.read_excel("../input/research_student (1).xlsx")


# In[82]:


train.head()


# In[6]:


train.info


# In[7]:


train.describe()


# In[8]:


train = train.drop([0,221,222])


# In[9]:


train.head()


# In[10]:


train.Branch.value_counts()


# In[13]:


train = train.fillna(0)


# In[12]:


train[['Branch','Gender']]


# In[14]:


train.fillna(0)


# In[ ]:


train.columns


# In[17]:


scale_list = [ 'Marks[10th]', 'Marks[12th]',
       'GPA 1', 'Rank', 'Normalized Rank', 'CGPA',
       'Current Back', 'Ever Back', 'GPA 2', 'GPA 3', 'GPA 4', 'GPA 5',
       'GPA 6', 'Olympiads Qualified', 'Technical Projects', 'Tech Quiz',
       'Engg. Coaching', 'NTSE Scholarships', 'Miscellany Tech Events']
sc = train[scale_list]
      


# In[18]:


sc.head()


# In[19]:


sc.tail()


# In[20]:


sc=sc.fillna(0)


# In[57]:


scaler = StandardScaler()
sc = scaler.fit_transform(sc)
train[scale_list] = sc
train[scale_list].head()


# In[58]:


train.head()


# In[22]:


train.info()


# In[ ]:





# In[59]:


encoding_list = ['Branch','Gender','Board[10th]','Board[12th]','Category']
train[encoding_list] = train[encoding_list].apply(LabelEncoder().fit_transform)


# In[60]:


train.head()


# In[26]:


train.info()


# **LINEAR REGRESSION**

# In[61]:


train.head()


# In[62]:


y = train['CGPA']
x = train.drop('CGPA', axis=1)


# In[29]:


x.info()


# In[63]:


y.info


# In[64]:


X_train, X_test, y_train, y_test = train_test_split(x, y ,test_size=0.3)


# In[65]:


X_train.shape


# In[66]:


X_test.shape


# In[67]:


logreg=LinearRegression()


# In[37]:


logreg.fit(X_train,y_train)


# In[ ]:





# In[68]:


y_pred=logreg.predict(X_test)


# In[39]:


y_test


# In[69]:


print(metrics.mean_squared_error(y_test, y_pred))


# In[50]:


xgb = xgboost.XGBRegressor(n_estimators=2500, learning_rate=0.06, gamma=0, subsample=0.6,
                           colsample_bytree=0.7, min_child_weight=4, max_depth=3)
xgb.fit(X_train,y_train)


# In[70]:


X_train.head()


# In[71]:


train.head()


# In[44]:


train.info()


# In[72]:


train.plot(kind="scatter",x="Marks[12th]",y="CGPA")


# In[73]:


train.plot(kind="scatter",x="Rank",y="CGPA")


# In[49]:


train.head()


# In[75]:


train.plot(kind="scatter",x="GPA 2",y="CGPA")


# In[76]:


train.plot(kind="scatter",x="GPA 4",y="CGPA")


# In[77]:


train.plot(kind="scatter",x="Marks[10th]",y="Marks[12th]")


# In[78]:


train


# In[81]:


a = np.random.random((16, 16))
plt.imshow(a, cmap="Marks[10th]", interpolation="CGPA")
plt.show()


# In[89]:


plt.hist("Marks[10th]",bins=56,histtype="bar",rwidth=0.8)


# In[ ]:





# In[ ]:





# In[ ]:




