#!/usr/bin/env python
# coding: utf-8

# **Exploring Titanic dataset and exploring about the people, who survived at that incident or not..**

# **Importing Needed Packages**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math


# In[ ]:


titanic_data= pd.read_csv("../input/titanic/train.csv")
titanic_data.head(10)


# In[ ]:


print("No. of passengers in original data : " +str(len(titanic_data.index)))


# **Analyzing Data**

# In[ ]:


sns.countplot(x="Survived", data=titanic_data)


# In[ ]:


sns.countplot(x="Survived", hue="Sex", data=titanic_data)


# In[ ]:


sns.countplot(x="Survived", hue="Pclass", data=titanic_data)


# In[ ]:


titanic_data["Age"].plot.hist()


# In[ ]:


titanic_data["Fare"].plot.hist(bins=20, figsize=(10,5))


# In[ ]:


titanic_data.info()


# In[ ]:


sns.countplot(x="SibSp", data=titanic_data)


# In[ ]:


sns.countplot(x="Parch", data=titanic_data)


# **Data Wrangling : **
# Removal of null values

# In[ ]:


titanic_data.isnull()


# In[ ]:


titanic_data.isnull().sum()


# In[ ]:


sns.boxplot(x="Pclass", y="Age", data=titanic_data)


# In[ ]:


titanic_data.head()


# In[ ]:


titanic_data.drop("Cabin", axis=1, inplace=True)


# In[ ]:


titanic_data.head()


# In[ ]:


titanic_data.dropna(inplace=True)


# In[ ]:


titanic_data.head()


# In[ ]:


titanic_data.isnull().sum()


# **For implementing Logistics regression, the string value should be converted to categorical values**

# In[ ]:


sex=pd.get_dummies(titanic_data["Sex"],drop_first=True)
sex.head()


# In[ ]:


embark=pd.get_dummies(titanic_data["Embarked"],drop_first=True)
embark.head()


# In[ ]:


titanic_data.drop(["Sex", "Embarked", "Name", "Ticket"], axis=1, inplace=True)


# In[ ]:


titanic_data=pd.concat([titanic_data,sex,embark], axis=1)


# In[ ]:


titanic_data.head()


# **Building a Logistics Regression Model**

# **Train Data**

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(titanic_data.drop('Survived',axis=1), titanic_data['Survived'], test_size=0.3, random_state=1)


# **Training and Predictions**

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel=LogisticRegression()


# In[ ]:


logmodel.fit(X_train, Y_train)


# In[ ]:


predictions=logmodel.predict(X_test)
X_test.head()


# In[ ]:


from sklearn.metrics import classification_report
classification_report(Y_test, predictions)


# In[ ]:


predictions


# **Evaluation**

# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, predictions)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(Y_test, predictions)

