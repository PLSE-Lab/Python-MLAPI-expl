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


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('/kaggle/input/health-care-data-set-on-heart-attack-possibility/heart.csv')


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.info()


# In[ ]:


df.describe().transpose()


# In[ ]:


df.isna().sum()


# In[ ]:


df['target'].value_counts()


# #  **Now since we finished importing and viewing information about the data, lets now do some data visualisation!!**

# In[ ]:


plt.figure(figsize=(10,6))
plt.style.use('ggplot')
sns.set_style('whitegrid')
sns.countplot(x='sex',data=df)


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')


# **As we can see, target is mostly dependent on chest pain(cp) and thalach**

# In[ ]:


df['cp'].value_counts()


# In[ ]:


df['thalach'].head(10)


# In[ ]:


plt.figure(figsize=(10,6))
sns.lineplot(x='target',y='cp',data=df)
plt.title('A line plot depicting the relation between target and chest pain')


# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x='cp',y='target',data=df,palette='Set1')
plt.title('A box plot depicting the relation between target and chest pain')


# In[ ]:


plt.figure(figsize=(8,4))
sns.violinplot(x='target',y='thalach',data=df)
plt.title('A violin plot depicting the relation between target and thalach')


# In[ ]:


plt.figure(figsize=(10,8))
sns.scatterplot(x='target',y='thalach',data=df,palette='Set2')
plt.title('A scatter plot depicting the relation between target and thalach')


# We can see that if the target is 1, there is more chest pain and more thalach
# Hence there is a good correlation among them

# In[ ]:


import cufflinks as cf
cf.go_offline()


# In[ ]:


df['age'].iplot(kind='hist',bins=25)


# In[ ]:


g=sns.countplot(x='target',data=df,hue='age',palette='Set1')
g.legend(loc='right', bbox_to_anchor=(1.25, 0.5), ncol=1)
plt.title('a plot depicting the ages versus target')


# In[ ]:


g=sns.countplot(x='target',hue='sex',data=df,palette='viridis')
g.legend(loc='right', bbox_to_anchor=(1.25, 0.5), ncol=1)


# Now since we have finished data visualisations, lets train our model using logistic regresssion!
# 

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X=df.drop('target',axis=1)
y=df['target']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lg=LogisticRegression()


# In[ ]:


lg.fit(X_train,y_train)


# In[ ]:


predictions=lg.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


print(confusion_matrix(y_test,predictions))


# In[ ]:


df['target'].value_counts()/len(df)*100


# We can see that there is an accuracy of 88. If our model only predicted all targets as 1, then we would have only 54% accuracy. So our model did a good job!!

# In[ ]:


print('Number of correct predictions are {}'.format(36+44))


# In[ ]:


print('Number of wrong predictions are {}'.format(8+3))


# Lets now take a random case and see whether our model predicts it correctly!

# In[ ]:


import random
random.seed(101)
num=random.randint(0,len(df))
num


# In[ ]:


new_patient=df.iloc[num]
new_patient


# In[ ]:


lg.fit(X_train,y_train)


# In[ ]:


pred=lg.predict(new_patient.drop('target').values.reshape(1,-1))


# In[ ]:


pred


# Our model predicted that the patient does'nt have heart attack which is true!
# THANK YOU!!!
# 

# In[ ]:




