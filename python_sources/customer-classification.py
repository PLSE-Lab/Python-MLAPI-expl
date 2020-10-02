#!/usr/bin/env python
# coding: utf-8

# **Santander Customer Satisfaction**
# 
# **Problem statement:-**
# 
# From frontline support teams to C-suites, customer satisfaction is a key measure of success. Unhappy customers don't stick around. What's more, unhappy customers rarely voice their dissatisfaction before leaving.
# 
# This kernel will help Santander Bank to identify dissatisfied customers early in their relationship. Doing so would allow Santander to take proactive steps to improve a customer's happiness before it's too late.
# 
# We are provided with an anonymized dataset containing a large number of numeric variables. The "TARGET" column is the variable to predict. It equals 1 for unsatisfied customers and 0 for satisfied customers.
# 
# The task is to predict the probability that each customer in the test set is an unsatisfied customer.
# 
# ![Customer Satisfaction](https://cdn.dribbble.com/users/489445/screenshots/1819359/pyxl_blog_creating_customer_experience_2.gif)
# 
# 

# # Import important libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn import metrics
from imblearn.over_sampling import SMOTE

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Lets load

# In[ ]:


train = pd.read_csv('../input/santander-customer-satisfaction/train.csv')
test = pd.read_csv('../input/santander-customer-satisfaction/test.csv')


# # Analysing dataset

# In[ ]:


train.head()


# In[ ]:


train.describe()


# We saw there are only numerical columns in this dataset. However looking at the range it seems there are many outliers here

# In[ ]:


train.isna().sum().sort_values(ascending=False)


# There are no null values present either
# 
# Lets check one of the column to confirm our theory about outliers

# In[ ]:


train['imp_ent_var16_ult1'].value_counts()


# Yes there it is . We can clearly see there is only 1 value of 17595.15 which definitely serve as an outlier. To clean this dataset outlier ommission is needed.

# # Outlier handling

# First we will see the values of different quartiles

# In[ ]:


# Checking outliers at 25%,50%,75%,90%,95% and 99%
train.describe(percentiles=[.25,.5,.75,.90,.95, .975,.99,.999])


# By analysing this visually it seems that ranges 0.99, 0.25, 0.75 seems important. Lets save their value somewhere

# In[ ]:


high = .99
first_quartile = 0.25
third_quartile = 0.75
quant_df = train.quantile([high, first_quartile, third_quartile])


# In[ ]:


quant_df


# Now we have our quartile dataframe

# Lets now prepare and clean our training dataset.
# 
# First we need to remove TARGET and ID column from our training dataset.

# In[ ]:


train_df = train.drop(['ID', 'TARGET'], axis = 1)


# #### Lets take 99% as threshold for outlier. Drop all values above 0.99 percentile

# In[ ]:


train_df = train_df.apply(lambda x: x[(x <= quant_df.loc[high,x.name])], axis=0)


# In[ ]:


train_df.describe(include='all')


# In[ ]:


train_df.shape


# In[ ]:


train_df.head()


# We can see that we didnt have that much data loss and also our data is now free of outliers.
# 
# Now lets get back our ID and TARGET columns

# In[ ]:


train_df = pd.concat([train.loc[:,'ID'], train_df], axis=1)

train_df = pd.concat([train.loc[:,'TARGET'], train_df], axis=1)


# In[ ]:


train_df.describe()


# In[ ]:


train_df.isnull().sum().sort_values(ascending=False)


# After dropping outliers we can see that we have now encountered some null values in our dataset. We need to cater this issue.
# 

# # Handling null values

# To handle null values we will take a random value between minimum and maximum value of each column and use that random value for imputation

# In[ ]:


new_train_df = train_df
for col in new_train_df.columns:
    min_val = min(new_train_df[col])
    max_val = max(new_train_df[col])
    new_train_df[col].fillna(round(random.uniform(min_val, max_val), 2), inplace =True)


# In[ ]:


new_train_df.isna().sum().sort_values(ascending=False)


# Now we can see our dataset is free of null values :)

# Now lets go to model testing

# In[ ]:


y = new_train_df['TARGET']
X = new_train_df.drop(['TARGET','ID'], axis=1)


# The much needed train test split for cross validation

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)


# # Model building

# In[ ]:


lr = LogisticRegression()
lr.fit(X_train,y_train)


# In[ ]:


preds = lr.predict(X_test)
print("Accuracy with Logistic = ", metrics.accuracy_score(y_test, preds))


# Accuracy with Logistic is quite good. But hold on do we have a balanced dataset. Maybe not. Lets check if imbalance is the root cause of this high accuracy.

# In[ ]:


new_train_df['TARGET'].value_counts()


# So we can see that almost 96% of dataset is having value 0. We need to balance this dataset to get better results

# ### Dataset balancing
# We will use **SMOTE** technique to balance the dataset

# In[ ]:


sm = SMOTE(kind = "regular")
X_tr,y_tr = sm.fit_sample(X_train,y_train)


# In[ ]:


print(X_tr.shape)
print(y_tr.shape)


# Lets now fit the data

# ### Logistic Regression

# In[ ]:


lr.fit(X_tr,y_tr)

lr_preds = lr.predict(X_test)
print("Accuracy with Logistic = ", metrics.accuracy_score(y_test, lr_preds))


# ## Decision tree

# In[ ]:


dt1 = DecisionTreeClassifier(max_depth=5)
dt1.fit(X_tr, y_tr)

dt_preds = dt1.predict(X_test)
print("Accuracy with Decision Tree = ", metrics.accuracy_score(y_test, dt_preds))


# ## Random Forest

# In[ ]:


rft = RandomForestClassifier(n_jobs=-1)
rft.fit(X_tr, y_tr)

rft_preds = rft.predict(X_test)
print("Accuracy with Random Forest = ", metrics.accuracy_score(y_test, rft_preds))


# We have got multiple models with all their accuracies. We can choose anyone of them to make further predictions.

# As RFT has maximum accuracy lets use that.

# # Final Prediction

# In[ ]:


x_test_final = test.drop(['ID'], axis=1)
final_prediction = rft.predict(x_test_final)


# In[ ]:


submission = pd.DataFrame({
        "ID": test["ID"],
        "TARGET": final_prediction
    })
submission.to_csv('RandomForect.csv',header=True, index=False)

