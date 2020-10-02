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


# # solving the scooter challenge

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


# In[ ]:


#loading the customer data, buy scooter and the test data
customer = pd.read_csv('/kaggle/input/dscmeetup3/CustomerInfo.csv')
buyscooter = pd.read_csv('/kaggle/input/dscmeetup3/buyscooter.csv')
testdata = pd.read_csv('/kaggle/input/dscmeetup3/testdata.csv')
test_copy = testdata.copy()


# In[ ]:


customer.head()


# In[ ]:


buyscooter.head()


# In[ ]:


testdata.head()


# In[ ]:



print('we have', customer.shape[0], 'rows and', customer.shape[1], 'columns in Customer Info')
print("==========================================================")
print('we have', buyscooter.shape[0], 'rows and', buyscooter.shape[1], 'columns in Buy scooter data')
print("==========================================================")
print('we have', testdata.shape[0], 'rows and', testdata.shape[1], 'columns in test data')


# # Joining buyscooter and customer data

# In[ ]:


merged_data = customer.merge(buyscooter)


# In[ ]:


merged_data.head()


# # Checking Point`

# In[ ]:


#Create checking point
df = merged_data.copy()


# In[ ]:


df.columns


# In[ ]:


df.describe().transpose()


# In[ ]:


df['Gender'] = df['Gender'].replace({'M': 'Male', 'F': 'Female'})


# In[ ]:


df.Gender.value_counts()


# In[ ]:



df['MaritalStatus'] = df.MaritalStatus.replace({'M': 'Married', 'S': 'Single'})


# In[ ]:


df.MaritalStatus.value_counts()


# In[ ]:


df.Education.value_counts()
#print('------------------------'),
#df.Occupation.value_counts()


# In[ ]:


object_data = df.dtypes[df.dtypes == 'object'].count()
categorical_data = df.dtypes[df.dtypes == 'int64'].count()
continuous_data = df.dtypes[df.dtypes == 'float64'].count()


# In[ ]:


print('we have {} object data'.format(object_data))
print('we have {} categorical data'.format(categorical_data))
print('we have {} continuous data'.format(continuous_data))


# In[ ]:


categorical_features = df.dtypes[df.dtypes == 'object'].index
continuous_features = df.dtypes[df.dtypes == 'int64'].index


# In[ ]:


# Counts on categorical columns
for feature in categorical_features:
    print(feature,':')
    print(df[feature].value_counts())
    print('----------------------------')


# In[ ]:


#Columns list
df.columns


# In[ ]:


# Columns to drop
target = df['BuyScooter']
train_test = [df, testdata]
drop_col = ['CustomerID', 'FirstName', 'MiddleName', 'LastName','City',
       'StateProvinceName','PostalCode', 'PhoneNumber']


# In[ ]:


for dataset in train_test:
    dataset.drop(drop_col, axis=1, inplace = True)


# In[ ]:


df.columns


# In[ ]:


df.dtypes


# In[ ]:


for dataset in train_test:
    dataset['BirthDate'] =  pd.to_datetime(dataset['BirthDate'])


# In[ ]:


df.dtypes


# In[ ]:


testdata.dtypes


# In[ ]:


testdata.head()


# # Feature Engineering

# In[ ]:


# Calculating Age


# In[ ]:


for dataset in train_test:
        dataset['TotalAsset'] = dataset['HomeOwnerFlag']+dataset['NumberCarsOwned']


# In[ ]:


for dataset in train_test:
    if (dataset['TotalChildren']==0).all():
        dataset['ChildIncomeR2'] = dataset['YearyIncome']/1.0
    else:
        dataset['ChildIncomeR2'] = dataset['YearlyIncome']/dataset['TotalChildren']


# - Education :
# - Bachelors              4191
# - Partial College        3905
# - High School            2580
# - Graduate Degree        2547
# - Partial High School    1181

# In[ ]:


df['Education'] = df['Education'].replace({'Partial High School': 1, 'High School':2, 'Partial College':3, 'Bachelors':4, 'Bachelors ':4,'Graduate Degree':5}) 
testdata['Education'] = testdata['Education'].replace({'Partial High School': 1, 'High School':2, 'Partial College':3, 'Bachelors':4, 'Bachelors ':4, 'Graduate Degree':5})
train = df.copy()
test = testdata.copy()


# In[ ]:


testdata.head()


# In[ ]:


df.TotalChildren.describe()


# In[ ]:


# from datetime import date
# def calculate_age(born):
#     today = datetime.date.today()
#     return today.year - born - (today.month, today.day) < (born.month, born.day)

# df['Age'] = df['BirthDate'].apply(calculate_age)

# # for dataset in train_test:
# #     dataset['Age'] = dataset['BirthDate'].apply(calculate_age)


# In[ ]:


categorical_features = df.dtypes[df.dtypes == 'object'].index
continuous_features = df.dtypes[df.dtypes == 'int64'].index


# In[ ]:


# Counts on categorical columns
for feature in categorical_features:
    print(feature,':')
    print(df[feature].value_counts())
    print('----------------------------')


# In[ ]:


df.Education.value_counts()


# In[ ]:


for dataset in train_test:
    dataset.drop(['BirthDate','ChildIncomeR2'], axis=1, inplace = True)


# In[ ]:


df.head()


# In[ ]:


testdata = pd.get_dummies(testdata)
df = pd.get_dummies(df)


# In[ ]:


df.head()


# In[ ]:


df.shape, testdata.shape


# In[ ]:


test.Education.value_counts()


# In[ ]:


test.columns


# In[ ]:


testdata.head()


# In[ ]:


features = df.drop('BuyScooter', axis=1)
target = df.BuyScooter


# In[ ]:


features.shape, testdata.shape


# # Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=.20, random_state=0)


# In[ ]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression


# In[ ]:


reg = LogisticRegression()
reg.fit(X_train, y_train)
print("Train score: ", reg.score(X_train, y_train))
print("Validation Score :",reg.score(X_test, y_test))


# ## Decision Tree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy", max_depth = 3)
dt.fit(X_train, y_train)
print("Train score: ", dt.score(X_train, y_train))
print("Validation Score :",dt.score(X_test, y_test))


# In[ ]:





# In[ ]:


merged_data.shape, 


# In[ ]:


prediction = reg.predict(testdata)

submission = pd.DataFrame({'CustomerID': test_copy['CustomerID'],
                          "BuyScooter": prediction})


# In[ ]:


# Decision Tree
prediction2 = dt.predict(testdata)

submission2 = pd.DataFrame({'CustomerID': test_copy['CustomerID'],
                          "BuyScooter": prediction2})
submission2.to_csv('Submission4.csv', index=False)


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('Submission.csv', index=False)


# In[ ]:


get_ipython().system('kaggle competitions submissions -c dscmeetup3 -f submission.csv -m "Notes"')


# In[ ]:




