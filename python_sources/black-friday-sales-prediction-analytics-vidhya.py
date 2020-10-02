#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
    


# In[ ]:


# Read train data
train = pd.read_csv('/kaggle/input/black-friday/train.csv')

# Read test data
test = pd.read_csv('/kaggle/input/black-friday/test.csv')


# **Having a look at train data**

# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


train.info()


# In[ ]:


train.isnull().sum()


# In[ ]:


train.dtypes


# In[ ]:


train.describe()


# **Having alook at test data**

# In[ ]:


test.head()


# In[ ]:


test.shape


# In[ ]:


test.info()


# **Missing values**

# In[ ]:


def missing(df):
    missing_values=df.isnull().sum()
    missing_percentage=missing_values*100/len(df['User_ID'])
    missing_percentage=missing_percentage.sort_values(ascending=False)
    return missing_percentage


# In[ ]:


missing(train)


# In[ ]:


missing(test)


# # EDA

# In[ ]:


train['Age'].value_counts()


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(train['Age'])


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(train['City_Category'])


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(train['Stay_In_Current_City_Years'])


# In[ ]:


train['Occupation'].unique()


# In[ ]:


train['City_Category'].unique()


# In[ ]:


train['Stay_In_Current_City_Years'].unique()


# In[ ]:


print(train['Product_Category_1'].unique())
print(train['Product_Category_2'].unique())
print(train['Product_Category_3'].unique())


# # Pre-processing****

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
train['User_ID'] = train['User_ID'] - 1000000
test['User_ID'] = test['User_ID'] - 1000000

enc = LabelEncoder()
train['User_ID'] = enc.fit_transform(train['User_ID'])
test['User_ID'] = enc.transform(test['User_ID'])


# In[ ]:


train['Product_ID'] = train['Product_ID'].str.replace('P00', '')
test['Product_ID'] = test['Product_ID'].str.replace('P00', '')

scaler = StandardScaler()
train['Product_ID'] = scaler.fit_transform(train['Product_ID'].values.reshape(-1, 1))
test['Product_ID'] = scaler.transform(test['Product_ID'].values.reshape(-1, 1))


# In[ ]:


categorical_col = ['Gender', 'City_Category']
numerical_col = ['Age', 'Occupation', 'Stay_In_Current_City_Years', 'Product_Category_1', 
           'Product_Category_2', 'Product_Category_3']


# In[ ]:


train['Age']=train['Age'].replace('0-17',17)
train['Age']=train['Age'].replace('18-25',25)
train['Age']=train['Age'].replace('26-35',35)
train['Age']=train['Age'].replace('36-45',45)
train['Age']=train['Age'].replace('46-50',50)
train['Age']=train['Age'].replace('51-55',55)
train['Age']=train['Age'].replace('55+',60)


# In[ ]:


test['Age']=test['Age'].replace('0-17',17)
test['Age']=test['Age'].replace('18-25',25)
test['Age']=test['Age'].replace('26-35',35)
test['Age']=test['Age'].replace('36-45',45)
test['Age']=test['Age'].replace('46-50',50)
test['Age']=test['Age'].replace('51-55',55)
test['Age']=test['Age'].replace('55+',60)


# In[ ]:


train['Stay_In_Current_City_Years']=train['Stay_In_Current_City_Years'].replace('4+',4)
test['Stay_In_Current_City_Years']=test['Stay_In_Current_City_Years'].replace('4+',4)


# **Filling missing values with zero**

# In[ ]:


train = train.fillna(0)
test = test.fillna(0)


# # Encoding

# In[ ]:


# Encoding categorical columns

encoder = LabelEncoder()

for col in categorical_col:
    train[col] = encoder.fit_transform(train[col])
    test[col] = encoder.transform(test[col])


# In[ ]:


# Scaling numerical columns

scaler = StandardScaler()

for col in numerical_col:
    train[col] = scaler.fit_transform(train[col].values.reshape(-1, 1))
    test[col] = scaler.transform(test[col].values.reshape(-1, 1))


# In[ ]:


train.head()


# # Training the model

# In[ ]:


X = train.drop(['Purchase'], axis=1)
y = train[['Purchase']]
X_test = test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)


# # 1.LinearRegression

# In[ ]:


reg=linear_model.LinearRegression()
lm_model=reg.fit(X_train,y_train)
pred=lm_model.predict(X_val)


# In[ ]:


np.sqrt(mean_squared_error(y_val,pred))


# # 2.Xg Boost

# In[ ]:


xgb_reg = XGBRegressor(learning_rate=1.0, max_depth=6, min_child_weight=40, seed=0)

xgb_reg.fit(X_train, y_train)
y_pred = xgb_reg.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_pred, y_val))

print (xgb_reg)


# In[ ]:


rmse


# **Choose xgboost over linear regression due to less RMSE **

# # If you find the kernel useful, upvote it :)

# 
