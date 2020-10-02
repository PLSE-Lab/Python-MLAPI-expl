#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style ('whitegrid')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#load data
data = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')


# In[ ]:


#check the head of the data
data.head()


# In[ ]:


data.info()


# **Data Preprocessing and Exploratory Data Analysis (EDA)**

# In[ ]:


#Drop customerID - not needed
data = data.drop(columns = 'customerID')
data.head()


# In[ ]:


#Convert TotalCharges to a numeric datatype
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors = 'coerce')
type(data['TotalCharges'][0])


# In[ ]:


#Check for missing values in each column
missing_values_count = data.isnull().sum()
missing_values_count


# In[ ]:


#TotalCharges is the only column with missing values
#fill missing values with the mean TotalCharges
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].mean())


# In[ ]:


#Let's check the missing values again to confirm that TotalCharges has no missing values
missing_values_count_2 = data.isnull().sum()
missing_values_count_2


# In[ ]:


#Convert SeniorCitizen from integer to string
data['SeniorCitizen'] = data['SeniorCitizen'].apply(lambda x: 'Yes' if x==1 else 'No')
data.head()


# In[ ]:


#Let's check the outcome variable Churn
sns.countplot(x='Churn', data = data)


# In[ ]:


sns.pairplot(data = data, hue = 'Churn')


# Insights from the above graph
# #- The customers with longer tenures general tend to stay

# In[ ]:


sns.heatmap(data.corr(),cmap = 'viridis', annot=True)


# Insights from the above graph
# #- TotalCharges is strongly correlated with tenure, which makes sense because you pay more with time
# #- It would be good to drop TotalCharges during modeling since it is strongly dependent on tenure and MonthlyCharges

# In[ ]:


plt.subplots(figsize=(12,4))
plt.subplot(1,3,1)
sns.boxplot(x='MonthlyCharges', y='Churn', data=data)
plt.subplot(1,3,2)
sns.boxplot(x='TotalCharges', y='Churn', data=data)
plt.subplot(1,3,3)
sns.boxplot(x='tenure', y='Churn', data=data)
plt.tight_layout()


# #Insights
# #- Customers with higher monthly charges have a higher churn rate
# #- Customers with longer tenure have lower churn rate

# In[ ]:


sns.countplot(x='Dependents', data=data, hue='Churn')


# In[ ]:


sns.countplot(x='Contract', data=data, hue='Churn')
#insight - month-to-month contract customers have a higher churn rate


# In[ ]:


sns.countplot(x='Partner', data=data, hue='Churn')


# In[ ]:


sns.countplot(x='InternetService', data=data, hue='Churn')
#Insight - customers with fiber optic service tend to have a higher churn rate


# In[ ]:


sns.countplot(x='TechSupport', data=data, hue='Churn')
#Insght - Customers without tech support tend to have a higher churn rate


# In[ ]:


sns.countplot(x='PaperlessBilling', data=data, hue='Churn')


# In[ ]:


data.sample(5)


# **Feature Engineering**
# **- Scaling of numeric variables**
# **- Getting dummies for categorical variables**

# In[ ]:


columns = list(data.columns)


# In[ ]:


numeric_cols = ['tenure', 'TotalCharges', 'MonthlyCharges']
non_numeric_cols = list(set(columns) - set(numeric_cols))
non_numeric_cols


# In[ ]:


non_numeric_data = pd.get_dummies(data[non_numeric_cols], drop_first=True)
non_numeric_data.sample(5)


# In[ ]:


from sklearn.preprocessing import scale
numeric_data = pd.DataFrame(scale(data[numeric_cols]), index=data.index, columns=numeric_cols)
numeric_data.head()


# In[ ]:


prepared_data = pd.concat([numeric_data, non_numeric_data], axis=1)
prepared_data.head()


# **Modeling**

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(prepared_data.drop(['TotalCharges', 'Churn_Yes'], axis=1),prepared_data['Churn_Yes'],test_size=0.30, random_state=42)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel=LogisticRegression()


# In[ ]:


logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


#Model Evaluation
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


print(confusion_matrix(y_test,predictions))


# In[ ]:


accuracy_score(y_test, predictions)


# 
