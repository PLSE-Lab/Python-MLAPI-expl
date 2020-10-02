#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #data visualization library
import seaborn as sns #data visualization library
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


ad= pd.read_csv('../input/Admission_Predict_Ver1.1.csv')


# In[ ]:


ad.head()


# In[ ]:


ad.columns


# In[ ]:


ad.info()


# # Visualizing data to understand the dataset

# In[ ]:


sns.countplot(x='Research', data=ad)


# In[ ]:


sns.boxplot(x='Research', y='Chance of Admit ', data=ad)


# **The number of students who has some kind of research paper is more than the number of students who don't have any research paper. Having a research paper increase the probability of getting accepted to a graduate school.**

# In[ ]:


sns.boxplot(x='University Rating', y='Chance of Admit ', data=ad)


# **Chance of getting an admission also change significantly based on university ranking.**

# In[ ]:


plt.figure(figsize=(10,8))
sns.scatterplot(x='GRE Score', y='TOEFL Score', size='Chance of Admit ', data=ad)


# **Hign GRE and TOEFL Score also increase students chance of Admission. A student who gets a good GRE score also tends to get a good TOEFL score. They are almost linearly related.**

# In[ ]:


sns.countplot(x='SOP', data=ad)


# In[ ]:


plt.figure(figsize=(12,8))
sns.boxplot(x='SOP', y='Chance of Admit ', data=ad)


# In[ ]:


sns.countplot(x='LOR ', data=ad)


# In[ ]:


plt.figure(figsize=(12,8))
sns.boxplot(x='LOR ', y='Chance of Admit ', data=ad)


# **A good letter of recommendation and SOP also assist in getting an admission. **

# In[ ]:


plt.figure(figsize=(8,8))
sns.scatterplot(x='CGPA', y='Chance of Admit ', data=ad)


# **CGPA also have a linear relation with the chance of admission.**

# In[ ]:


plt.figure(figsize=(7,6))
sns.heatmap(ad.corr(),annot=True, cmap='magma')


# ## After examing relation between the chance of admission and every other feature it is evident that the chance of admission has a linear relation with every other feature. Hence, the Linear Regression model will be a great choice to predict Admission chance of a student.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


ad.drop('Serial No.', axis=1, inplace=True)


# In[ ]:


X= ad.drop('Chance of Admit ', axis=1)
y=ad['Chance of Admit ']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=105)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lreg= LinearRegression()


# In[ ]:


lreg.fit(X_train, y_train)


# In[ ]:


pred= lreg.predict(X_test)


# In[ ]:


pred[:10]


# In[ ]:


y_test.head(10)


# In[ ]:


from sklearn import metrics


# In[ ]:


print('Mean Squared Error: ',metrics.mean_squared_error(y_test,pred))
print('Root Mean Squared Error: ',np.sqrt(metrics.mean_squared_error(y_test,pred)))

