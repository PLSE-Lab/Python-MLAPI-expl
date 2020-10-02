#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/diabetes2.csv')


# In[ ]:


df


# **Data is clean no need to preprocess it**

# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# **From this we can deduct  Gluecose BMI AGE and PREGRANCIES have effect on diabetic condition**

# In[ ]:


sns.heatmap(df.corr(),annot=True)


# In[ ]:


sns.scatterplot('Glucose','BMI',hue='Outcome',data=df)


# In[ ]:


sns.pairplot(df,hue='Outcome')


# In[ ]:


X = df.iloc[:,0:8]


# In[ ]:


X


# In[ ]:


y=df.iloc[:,-1:]


# In[ ]:


y


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lr = LogisticRegression()


# In[ ]:


lr.fit(X_train,y_train)


# In[ ]:


predict = lr.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(classification_report(y_test,predict))


# In[ ]:


cm = confusion_matrix(y_test,predict)


# In[ ]:


sns.heatmap(cm,annot=True,cmap='viridis')

