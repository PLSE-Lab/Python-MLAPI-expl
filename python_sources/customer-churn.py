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


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Load data & observe the features

# In[ ]:


df = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')


# In[ ]:


df.info()


# In[ ]:


df.describe().T


# In[ ]:


df.head()


# In[ ]:


df['TotalCharges'] =  pd.to_numeric(df['TotalCharges'],errors='coerce')


# Find out if there are any null values

# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# EDA and observe

# In[ ]:


sns.countplot(x='Churn',data=df)


# In[ ]:


sns.countplot(x='Churn',hue='PaymentMethod',data=df,palette='coolwarm')


# In[ ]:


sns.countplot(x='Churn',hue='gender',data=df,palette='coolwarm')


# In[ ]:


sns.countplot(x='Churn',hue='SeniorCitizen',data=df,palette='coolwarm')


# In[ ]:


sns.pairplot(data=df,palette='coolwarm')


# In[ ]:


sns.factorplot(x='Churn',y="tenure",data=df,kind="violin",palette="muted")


# In[ ]:


sns.factorplot(x='Churn',y="MonthlyCharges",data=df,kind="violin",palette="muted")


# Feature Engineering

# In[ ]:


target = ['Churn']


# In[ ]:


cat_cols = df.nunique()[df.nunique() < 6].keys().tolist()


# In[ ]:


bin_cols = df.nunique()[df.nunique() == 2].keys().tolist()


# In[ ]:


multi_cols= [i for i in cat_cols if i not in bin_cols + target]       


# Label Encode the categorical features and Scale the continous features

# In[ ]:


# Dropping missing values|

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# In[ ]:


le = LabelEncoder()
for i in bin_cols :
    df[i] = le.fit_transform(df[i])


# In[ ]:


#Duplicating columns for multi value columns
df = pd.get_dummies(data = df ,columns = multi_cols)


# In[ ]:


df.info()


# In[ ]:


num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']


# In[ ]:


df[num_cols].info()


# In[ ]:


df.drop(['customerID'],axis=1,inplace=True)


# In[ ]:


ss = StandardScaler()


# In[ ]:


scaled = ss.fit_transform(df[num_cols])


# In[ ]:


scaled = pd.DataFrame(data=scaled,columns=num_cols)


# In[ ]:


scaled.head()


# In[ ]:


scaled.info()


# Drop original columns

# In[ ]:


df.drop(columns = num_cols,axis =1,inplace=True)


# In[ ]:


df = pd.merge(df,scaled,left_index=True,right_index=True,how='left')


# In[ ]:


df.info()


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.head()


# In[ ]:


print(len(df.columns))
df.columns


# Modelling using Logistic Regression

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df.drop(['Churn'],axis=1),df['Churn'],test_size=0.3,random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

model = LogisticRegression()
model.fit(X_train,y_train)

predictions = model.predict(X_test)
score = model.score(X_test,y_test)

print("Accuracy =" + str(score))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# Most Important Features

# In[ ]:


arr_coeff= np.reshape(abs(model.coef_),(40,1))


# In[ ]:


column_name = ['Coeff']


# In[ ]:


coeff = pd.DataFrame(data=arr_coeff,index=X_test.columns.values,columns=column_name)


# In[ ]:


coeff.sort_values(by='Coeff',ascending=False)


# Compare with KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


pred = knn.predict(X_test)


# In[ ]:


score = model.score(X_test,y_test)
print("Accuracy =" + str(score*100))
print(confusion_matrix(pred,y_test))
print(classification_report(pred,y_test))


# Find Best K

# In[ ]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# Fitting again with k = 8 (low value)

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
score = model.score(X_test,y_test)
print("Accuracy =" + str(score*100))
print(confusion_matrix(pred,y_test))
print(classification_report(pred,y_test))


# Precision and recall have improved

# In[ ]:




