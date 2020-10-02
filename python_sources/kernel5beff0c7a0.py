#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df= pd.read_csv('../input/Admission_Predict.csv')


# In[4]:


df.head()


# In[5]:


df.describe()


# In[7]:


#getting information about the data set
df.info()


# In[10]:


#check null values
# we cansee from info that there are no null values in the data set but if there are any null values we can visualize them with seaborn
sns.heatmap(df.isnull(), cmap= 'viridis')


# In[11]:


#since data set is relatively small we can visualize how each column is realated to other columns in the data set.
df.drop('Serial No.', axis = 1, inplace=True)


# In[12]:


sns.set_style('whitegrid')
sns.pairplot(data = df, palette= 'rainbow')


# In[13]:


#from above pair plot we can see that Chance of Admit share linear realation with GRE Score, TOEFL score  and CGPA
# we confirm this with following joint plots
features  = ['GRE Score', 'TOEFL Score', 'CGPA']
for fea in features:
    sns.jointplot(df[(fea)], y = df[('Chance of Admit ')],kind ='reg')


# In[14]:


# lets explore data more
for fea in features:
    plt.figure(figsize=(10,4))
    sns.distplot(df[fea], color= 'red', kde = False, bins = 50)


# In[15]:


features2 = ['University Rating', 'SOP', 'LOR ', 'Research']
for fea in features2:
    sns.jointplot(df[(fea)], y = df['Chance of Admit '],kind ='hex')


# In[16]:


#checking the correlation
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, linewidths=0.05, fmt= '.2f',cmap="plasma",)


# In[17]:


df.loc[df['Chance of Admit ']>=0.9, 'Target'] = "Highly Likely"
df.loc[(df['Chance of Admit ']>=0.8) &(df['Chance of Admit ']<0.9), 'Target'] = "Likely"
df.loc[(df['Chance of Admit ']>=0.7) &(df['Chance of Admit ']<0.8), 'Target'] = "Reach"
df.loc[df['Chance of Admit ']<.7, 'Target'] = "Not Possible"


# In[18]:


df.head()


# In[19]:


from sklearn.preprocessing import LabelEncoder


# In[20]:


lbl_enc = LabelEncoder()
df['Target_encoder'] = lbl_enc.fit_transform(df['Target'])


# In[21]:


df.info()


# In[22]:


#lets predict chances
X = df.drop(['Chance of Admit ', 'Target'], axis=1)
y = df['Target_encoder']


# In[23]:


X.head()


# In[24]:


y.head()


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[27]:


#lets predict using logistic Regression
from sklearn.linear_model import LogisticRegression


# In[28]:


log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)


# In[29]:


log_prediction = log_reg.predict(X_test)


# In[30]:


from sklearn.metrics import classification_report, confusion_matrix


# In[31]:


print(classification_report(y_test, log_prediction))


# In[ ]:




