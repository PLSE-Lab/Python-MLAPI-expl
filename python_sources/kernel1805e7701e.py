#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


#vizualisation
import seaborn as sns
sns.pairplot(df,hue='Loan_Status')


# In[ ]:


sns.catplot(x='ApplicantIncome', y = 'Loan_Status', data = df)


# In[ ]:


sns.catplot(x='CoapplicantIncome', y = 'Loan_Status', data = df)


# In[ ]:


sns.catplot(x='LoanAmount', y = 'Loan_Status', data = df)


# In[ ]:


#preprocessing 
#check for null values
df.isnull().sum()


# In[ ]:


sns.countplot(df['Gender'])


# In[ ]:


#since male numbers are dominant therefore lets make null values as male
df['Gender'].fillna('Male',inplace = True)

df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace= True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(),inplace = True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace = True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0] , inplace = True)


# In[ ]:


sns.countplot(df['Self_Employed'])


# In[ ]:


df['Self_Employed'].fillna('No',inplace = True)


# In[ ]:


df.isnull().sum()


# In[ ]:


#preprocessing
X = df.iloc[:,[1,3,4,5,6,7,8,9,10,11]].values
Y = df.iloc[:,12].values


# In[ ]:


df.drop(columns=['Loan_ID','Married'],inplace = True)


# In[ ]:


df


# In[ ]:


#getting dummies
df = pd.get_dummies(df)
df.drop(columns=['Gender_Male','Dependents_0','Education_Not Graduate','Self_Employed_Yes','Property_Area_Urban','Loan_Status_N'],inplace = True)
df


# In[ ]:


X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values


# In[ ]:



##Feature Scaling


# In[ ]:



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(X)


# In[ ]:


#Splitting
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.25)


# In[ ]:


#Model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='gini')
classifier.fit(X_train,y_train)


# In[ ]:


#prediction
y_predict = classifier.predict(X_test)


# In[ ]:


#connfusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predict)
cm


# In[ ]:


print('accuracy is ',(29+83)/(29+83+19+23) *100,'%')


# In[ ]:




