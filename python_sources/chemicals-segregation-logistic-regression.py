#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dfx = pd.read_csv('../input/Logistic_X_Train.csv')
dfy = pd.read_csv('../input/Logistic_Y_Train.csv')
xtest = pd.read_csv('../input/Logistic_X_Test.csv')


# In[3]:


dfx.head()


# In[4]:


dfy.head()


# In[5]:


df=dfx.join(dfy)


# In[6]:


df.head()


# In[7]:


df = df.iloc[0:1000]


# In[8]:


df.shape


# In[9]:


df.info()


# In[10]:


x = df.iloc[0:,0:3].values
y = df.iloc[0:,3].values


# In[11]:


print(x)


# In[12]:


x.shape


# In[13]:


y.shape


# ### Baseline model of logistic regression
# 

# In[14]:


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x)
x_test = sc.fit_transform(xtest)


# In[15]:


x_test


# In[16]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y)
y_pred = logreg.predict(x_test)
acc_log = round(logreg.score(x_train, y) * 100, 2)
acc_log


# In[17]:


y_pred = y_pred.astype(int)


# In[18]:


np.savetxt('Predicted Solution.csv',y_pred)


# In[19]:


y_pred


# In[20]:


# Support Vector Machines
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y)
y_pred = svc.predict(x_test)
acc_svc = round(svc.score(x_train, y) * 100, 2)
acc_svc


# In[21]:


# Linear SVC
from sklearn.svm import LinearSVC
linear_svc = LinearSVC()
linear_svc.fit(x_train, y)
y_pred = linear_svc.predict(x_test)
acc_linear_svc = round(linear_svc.score(x_train, y) * 100, 2)
acc_linear_svc


# In[22]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(x_train, y)
y_pred = gaussian.predict(x_test)
acc_gaussian = round(gaussian.score(x_train, y) * 100, 2)
acc_gaussian


# In[23]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y)
y_pred = random_forest.predict(x_test)
random_forest.score(x_train, y)
acc_random_forest = round(random_forest.score(x_train, y) * 100, 2)
acc_random_forest

