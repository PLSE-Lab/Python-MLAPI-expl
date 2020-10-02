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

# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train = pd.read_csv('/kaggle/input/titanic/train.csv')


# ** Inspecting the dataframe**

# In[ ]:


df_train.head()


# In[ ]:


df_train.info()


# **Checking the Missing Values in Data Frame**

# In[ ]:


#Checking the % of Null Values

round(df_train.isna().sum() / len(df_train.index)*100,2)


# ## It is clear that Age is having around 20% Missing values and Cabin is having around 77%

# ## Impute the Age column with  mean value of Age

# In[ ]:


df_train['Age'].fillna((df_train['Age'].mean()), inplace=True)


# ## Some Charts for EDA

# In[ ]:


plt.figure(figsize = (10,7))
sns.countplot(x='Pclass',hue = 'Survived',data=df_train)


# ## It is clear that the highest mortality is amongst the passengers belonging to Class 3

# In[ ]:


plt.figure(figsize = (10,7))
sns.countplot(x='Sex',hue = 'Survived',data=df_train)


# In[ ]:


counts, bins = np.histogram(df_train['Age'])
plt.hist(bins[:-1], bins, weights=counts,histtype = 'bar',color = 'magenta')


# ## Maximum people on the ship was beloinging to Age Group 20-30

# ## Data Preparation

# In[ ]:


#Cabin column has lot of null values and its not serving any purpose for our analysis to dropping it from the dataframe
df_train = df_train.drop('Cabin',axis=1)


# In[ ]:


df_train.dropna()


# ## Creating Dummy columns for Sex  and Embarked Columns

# In[ ]:


m_sex = pd.get_dummies(df_train['Sex'],drop_first=True)
m_embarked = pd.get_dummies(df_train['Embarked'],drop_first=True)


# In[ ]:


df_train = df_train.drop(['Name','Sex','Ticket','Embarked'] , axis=1)


# In[ ]:


# Concatinate main dataframe and dummy columns
df_train = pd.concat([df_train,m_sex,m_embarked],axis=1)


# ## Model Building

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


# Putting feature variable to X
X = df_train.drop(['Survived'], axis=1)

X.head()


# In[ ]:


# Putting response variable to y
y = df_train['Survived']

y.head()


# In[ ]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# ## Feature Scaling to bring all columns to same scale

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train[['Age','Fare']] = scaler.fit_transform(X_train[['Age','Fare']])

X_train.head()


# In[ ]:


from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(X_train,y_train)


# ## Make Prediction on Train Set

# In[ ]:


pred = lm.predict(X_test)
X_test.head()


# In[ ]:


pred


# ## Import the Test Data Set

# In[ ]:


df_test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


df_test.head()


# ## Perform same opertions on test data as we performed on train data set

# In[ ]:


df_test['Age'].fillna((df_test['Age'].mean()), inplace=True)


# In[ ]:


df_test = df_test.drop('Cabin',axis=1)


# In[ ]:


df_test.dropna()


# In[ ]:


m_sex_test = pd.get_dummies(df_test['Sex'],drop_first=True)
m_embarked_test = pd.get_dummies(df_test['Embarked'],drop_first=True)


# In[ ]:


df_test = df_test.drop(['Name','Sex','Ticket','Embarked'] , axis=1)


# In[ ]:


# Concatinate main dataframe and dummy columns
df_test = pd.concat([df_test,m_sex_test,m_embarked_test],axis=1)


# In[ ]:


df_test['Fare'].fillna((df_test['Fare'].mean()), inplace=True)


# In[ ]:


test_pred = lm.predict(df_test)


# In[ ]:


test_pred.shape


# In[ ]:


df_test_predicted = pd.DataFrame(test_pred, columns= ['Survived'])


# In[ ]:


df_test_new = pd.concat([df_test,df_test_predicted],axis=1 , join = 'inner')


# In[ ]:


df_final = df_test_new[['PassengerId','Survived']]


# In[ ]:


df_final.to_csv('titanic_pred_final.csv' , index=False)


# In[ ]:




