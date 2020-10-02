#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv("../input/creditcard.csv")


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df["Time"].describe()


# In[8]:


df["Time"].unique()  # time in seconds since the first transaction in dataset


# In[9]:


df["Class"].unique()


# In[10]:


reals = df[df["Class"]==0];
frauds = df[df["Class"]==1];


# In[11]:


# randomly sample some fields and see what they look like
rand_reals_v1 = reals["V1"].sample(n=100);
rand_fakes_v1 = frauds["V1"].sample(n=100);


# In[12]:


plt.subplot(1,2,1);
plt.scatter(x=np.linspace(1,100,100),y=rand_reals_v1);
plt.subplot(1,2,2);
plt.scatter(x=np.linspace(1,100,100),y=rand_fakes_v1);


# In[13]:


rand_reals_v1.describe()


# In[14]:


rand_fakes_v1.describe()


# In[15]:


df2 = df[["V1", "V2", "V3", "V4", "V5"]].sample(n=1000)
pd.plotting.scatter_matrix(df2,figsize=(12,12))


# In[16]:


groups = df.groupby(df["Class"])


# In[17]:


fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.V1, group.V2, marker='o', linestyle='', ms=12, label=name)
ax.legend()


# In[18]:


fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.V1, group.V3, marker='o', linestyle='', ms=12, label=name)
ax.legend()


# In[19]:


fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.V1, group.V4, marker='o', linestyle='', ms=12, label=name)
ax.legend()


# In[20]:


fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.V1, group.V5, marker='o', linestyle='', ms=12, label=name)
ax.legend()


# In[21]:


fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.V1, group.V6, marker='o', linestyle='', ms=12, label=name)
ax.legend()


# In[22]:


fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.V1, group.V7, marker='o', linestyle='', ms=12, label=name)
ax.legend()


# In[23]:


import sklearn.utils


# In[24]:


df[df["Class"]==1]["V1"].count()


# In[25]:


df[df["Class"]==0]["V1"].count()


# In[26]:


# try naive upsampling (probably not the wisest)
nToSample = df[df["Class"]==0]["V1"].count() - df[df["Class"]==1]["V1"].count();
frauds = sklearn.utils.resample(df[df["Class"]==1],n_samples=nToSample)


# In[27]:


frauds["V1"].count()


# In[28]:


reals = df[df["Class"]==0]


# In[29]:


reals["V1"].count()


# In[30]:


df_upsampled = pd.concat([reals, frauds])
df_upsampled.head()


# In[32]:


# verify we upsampled well
print(df_upsampled[df_upsampled["Class"]==1]["V1"].count())
print(df_upsampled[df_upsampled["Class"]==0]["V1"].count())


# In[33]:


import sklearn.model_selection


# In[34]:


del df_upsampled["Time"]


# In[35]:


df_upsampled.head()


# In[36]:


df_upsampled.iloc[0:2, 0:27]


# In[37]:


df_model_train = df_upsampled.sample(frac=0.1)


# In[38]:


X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(df_model_train.iloc[:,0:27],df_model_train["Class"],test_size=0.2)


# In[40]:


SVC = sklearn.svm.LinearSVC()
SVC.fit(X_train,Y_train)


# In[41]:


SVC.score(X_test,Y_test)


# In[42]:


from sklearn.model_selection import GridSearchCV


# In[43]:


param_grid = {'penalty': ['l2'],
              'loss':['hinge','squared_hinge'],
              'C':[1, 10, 100, 1000]}


# In[44]:


SVC2 = sklearn.svm.LinearSVC()
clf = GridSearchCV(SVC2,param_grid)


# In[45]:


clf.fit(X_train,Y_train)


# In[46]:


clf.score(X_test, Y_test)


# In[47]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier


# In[48]:


clf = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, X_train, Y_train)


# In[49]:


scores.mean()


# In[50]:


clf.fit(X_train,Y_train)


# In[51]:


clf.score(X_test,Y_test)


# In[ ]:




