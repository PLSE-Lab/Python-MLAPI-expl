#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# **Importing Libraries**

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
import statsmodels.formula.api as smf


# **Loading Data**

# In[ ]:


df = pd.read_csv('../input/FIFA 2018 Statistics.csv')


# **Data Exploration**

# In[ ]:


print ('There are',len(df.columns),'columns:')
for x in df.columns:
    print(x,end=',')


# In[ ]:


df.head()


# **No null data**

# In[ ]:


df.info()


# Descriptive statistics , analysis of  both numeric and object series . 
# Produces elucidating insights that condense the focal propensity, scattering and state of a dataset's distribution, excluding NaN values.
# 

# In[ ]:


df.describe()


# Using heatmap graphical representation of Seaborn Library to check whether there is a missing data.
# Eventually, we found that 1st goal column misses many rows whereas Own Goal and Own Goal Time columns misses some of the data so to drop them in their visualization and prediction of Man of the Match column is fine.

# In[ ]:


plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


sns.jointplot(x='Attempts',y='Goal Scored',data=df,kind='kde')


# In[ ]:


sns.distplot(df['Goal Scored'],kde=False,axlabel='Goals scored in a match')


# In[ ]:


sns.jointplot(x='Ball Possession %',y='Goal Scored',data=df,kind='kde')


# In[ ]:





# In[ ]:


plt.figure(figsize=(10,10))
sns.countplot(y='Saves',data=df)


# In[ ]:


sns.countplot(y='Goal Scored',data=df)


# In[ ]:


plt.figure(figsize=(10,10))
sns.countplot(y='Fouls Committed',data=df)


# Converting categorical variable** (**in this case **Round** and **Man of the Match categorical variables** **)**into dummy/indicator variables and concating them in the DataFrame df.

# In[ ]:


df = pd.concat([df,pd.get_dummies(df['Round'])],axis=1)


# In[ ]:


mom = pd.get_dummies(df['Man of the Match'],drop_first=True)
df = pd.concat([df,mom, pd.get_dummies(df['Round'])],axis=1)


# **Importing Libraries**

# In[ ]:


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


X = df.drop(['Date','Team','Opponent','Round','PSO','Man of the Match','1st Goal','Own goals','Own goal Time'],axis=1)
y = df['Man of the Match']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


#    Creating a Logistic Regression model beacause we are predicting Man of the Match column and its a categorical series.

# In[ ]:


lr = LogisticRegression()


# Fitting Training data set to the Logistic Regression model lr

# In[ ]:


lr.fit(X_train,y_train)


# Creating predictions by passing Training dataset X_test to lr using predict() function.

# In[ ]:


predictions = lr.predict(X_test)


# In[ ]:


print (classification_report(y_test,predictions))


# Our trained model has made all the right predictions giving 100% accuracy.

# In[ ]:


print('Feature Coefficients of the Regression:- \n', lr.coef_[0])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




