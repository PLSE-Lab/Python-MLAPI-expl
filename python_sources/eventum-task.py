#!/usr/bin/env python
# coding: utf-8

# # Airbnb New User Bookings

# ## Import Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Loading The DataSet

# In[ ]:


# Load the data into DataFrames
train_users = pd.read_csv('../input/airbnb-recruiting-new-user-bookings/train_users_2.csv')
test_users = pd.read_csv('../input/airbnb-recruiting-new-user-bookings/test_users.csv')


# In[ ]:


print("Number of users in training set =", train_users.shape[0] )
print("Number of users in test set =",test_users.shape[0])


# In[ ]:


train_users.head()


# In[ ]:


train_users.describe(include = 'all')


# In[ ]:


test_users.head()


# In[ ]:


test_users.describe(include = 'all')


# From above, We can see that date_first_booking feature is allways NaN in test dataset so I will remove it from both training and testing.

# ## Data Cleaning

# In[ ]:


labels = train_users['country_destination'].values
train_users = train_users.drop(['country_destination', 'date_first_booking'], axis=1)
test_users = test_users.drop(['date_first_booking'], axis=1)
id_test = test_users['id']

# Merge train and test users
all_users = pd.concat((train_users, test_users), axis=0, ignore_index=True)

# Remove ID's since now we are not interested in making predictions
all_users.drop('id',axis=1, inplace=True)

all_users.head()


# In[ ]:


from datetime import datetime
all_users['date_account_created'] = pd.to_datetime(all_users['date_account_created'])
all_users['timestamp_first_active'] = pd.to_datetime((all_users.timestamp_first_active // 1000000), format='%Y%m%d')

all_users['date_account_created'] = [datetime.timestamp(d) for d in all_users['date_account_created']]
all_users['timestamp_first_active'] = [datetime.timestamp(d) for d in all_users['timestamp_first_active']]


# In[ ]:


all_users.age.describe()


# In[ ]:


sns.distplot(all_users.age.dropna())
plt.xlabel('Age')


# As we can see above, the common age to travel is between 14 and 70. So I will smooth Age distribution by remove all values bellow 14 and above 70.

# In[ ]:


sns.distplot(all_users.age.loc[all_users['age'] < 70].dropna())
plt.xlabel('Age')


# In[ ]:


all_users['age'] = np.where(all_users['age']<=14, 14, all_users['age'])
all_users['age'] = np.where(all_users['age']>=70, 70, all_users['age'])
all_users['age'] = all_users['age'].fillna(all_users['age'].dropna().values.mean())
all_users['age'].describe()


# In[ ]:


all_users['age'].values.mean()


# ## Feature Engineering

# In[ ]:


categorical_features = [
    'affiliate_channel',
    'affiliate_provider',
    'first_affiliate_tracked',
    'first_browser',
    'first_device_type',
    'gender',
    'language',
    'signup_app',
    'signup_method'
]

# one-hot-encoding
for categorical_feature in categorical_features:
    all_users_dummies = pd.get_dummies(all_users[categorical_feature], prefix=categorical_feature)
    all_users = all_users.drop([categorical_feature], axis=1)
    all_users = pd.concat((all_users, all_users_dummies), axis=1)


# In[ ]:


all_users.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

train_users_n = train_users.shape[0]
X_train = all_users.values[:train_users_n]
le = LabelEncoder()
y_train = le.fit_transform(labels)   
X_test = all_users.values[train_users_n:]


# ## Classification

# In[ ]:


def generate_answer(y_pred, classifer_name):
    #Taking the 5 classes with highest probabilities
    ids = []  #list of ids
    cts = []  #list of countries
    for i in range(len(id_test)):
        idx = id_test[i]
        ids += [idx] * 5
        cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()
    
    sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
    sub.to_csv(classifer_name+'.csv',index=False)
    return sub


# ### MLP Classifier

# In[ ]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier()
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict_proba(X_test)
generate_answer(y_pred_mlp, 'MLP')


# ### XGBClassifier

# In[ ]:


from xgboost.sklearn import XGBClassifier

xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)                  
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict_proba(X_test)
generate_answer(y_pred_xgb, 'XGB')


# In[ ]:




