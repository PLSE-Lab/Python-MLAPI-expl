#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression To predict Titanic Deaths

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


train = pd.read_csv("../input/train.csv")

train.head()


# 
# ## DATA ANALYSIS

# ### Missing Data
# 
# first we have to deal with missing data.

# In[ ]:


sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level. We'll probably drop this later, or change it to another feature like "Cabin Known: 1 or 0"

# ### Analysing The Surving Count

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)


# In[ ]:


sns.countplot(x = 'Survived', hue = 'Sex', data = train)


# In[ ]:


sns.countplot(x='Survived',hue='Pclass',data=train)


# ### Analysing Age Chart

# In[ ]:


sns.distplot(train['Age'].dropna(),kde=False,bins=40)


# In[ ]:


sns.countplot(x='SibSp',data=train)


# In[ ]:


sns.distplot(train['Fare'],kde=False,bins=40, color = 'darkgreen')


# ### Data Cleaning
# 
# We want to fill in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation). However we can be smarter about this and check the average age by passenger class and Sex.

# In[ ]:


plt.figure(figsize = (10,6))

sns.boxplot(x = 'Pclass', y = 'Age', data = train)


# In[ ]:


# define average age according to Pclass

def defage(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[ ]:


train['Age'] = train[['Age','Pclass']].apply(defage,axis=1)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False)


# In[ ]:


train.head()


# In[ ]:


train.drop('Cabin',axis=1,inplace=True)


# In[ ]:


# now get_dummies used to remove categorical variables 

sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[ ]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


train = pd.concat([train,sex,embark],axis=1)


# In[ ]:


train.head()


# ## Building a Logistic Regression Model

# ### Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# ### Trainning and Predicting

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# ### Evaluation
# 
# We can check precision,recall,f1-score using classification report!

# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:




