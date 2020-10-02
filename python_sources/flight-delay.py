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
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/flight-delays-fall-2018/flight_delays_train.csv.zip", compression='zip')
test = pd.read_csv("../input/flight-delays-fall-2018/flight_delays_test.csv.zip", compression='zip')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.info()
print('-'*45)
test.info()


# In[ ]:


train.describe()


# In[ ]:


# list of columns with missing values apart from target within test set
train.columns[train.isna().any()]


# In[ ]:


all_data = pd.concat([train, test], ignore_index=True)


# In[ ]:


# change target name to make it easier
train = train.rename(columns={'dep_delayed_15min':'delayed'})
all_data = all_data.rename(columns={'dep_delayed_15min':'delayed'})


# In[ ]:


# change target to numerical N-->0 & Y-->1
train.loc[(train.delayed == 'N'), 'delayed'] = 0
train.loc[(train.delayed == 'Y'), 'delayed'] = 1
all_data.loc[(all_data.delayed == 'N'), 'delayed'] = 0
all_data.loc[(all_data.delayed == 'Y'), 'delayed'] = 1


# # Exploratory Data Analysis & Data Cleaning

# ## 1- Month

# * Let's first change the format of the variable to an int

# In[ ]:


train.Month = train.Month.str.slice(start=2).astype(int)
all_data.Month = all_data.Month.str.slice(start=2).astype(int)


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(18,5))
sns.countplot('Month', data=train, ax=ax[0])
ax[0].set_title('Nb of flights by month')
sns.countplot('Month', hue='delayed', data=train, ax=ax[1])
ax[1].set_title('Delayed/Not delayed flights by month')
plt.figure(figsize=(18,5))
sns.barplot('Month', 'delayed', data=train)
plt.show()


# * We can see that the number of flights and delays are pretty much the same for all months. However there is a slightly higher rate of delays in June, July and December maybe due to vacation time
# * We will also try to reformat this variable into 1-12 instead of c1-c12 and int type

# ## 2- Day of Month

# * Again, let's first format the variable

# In[ ]:


train.DayofMonth = train.DayofMonth.str.slice(start=2).astype(int)
all_data.DayofMonth = all_data.DayofMonth.str.slice(start=2).astype(int)


# In[ ]:


fig, ax = plt.subplots(3, 1, figsize=(15,15))
sns.countplot('DayofMonth', data=train, ax=ax[0])
ax[0].set_title('Nb of flights by day of month')
sns.countplot('DayofMonth', hue='delayed', data=train, ax=ax[1])
ax[1].set_title('Delayed/not Delayed flight by day of month')
sns.barplot('DayofMonth', 'delayed', data=train, ax=ax[2])
ax[2].set_title('Rate of delayed flights by day of month')


# * Again it's hard to say if there is much of a difference between the days of the month. However, we can say that in the last days of the month the delay rate is higher

# ## 3- Day of Week

# * First let's format the variable

# In[ ]:


train.DayOfWeek = train.DayOfWeek.str.slice(start=2).astype(int)
all_data.DayOfWeek = all_data.DayOfWeek.str.slice(start=2).astype(int)


# In[ ]:


fig, ax = plt.subplots(1, 3, figsize=(15,5))
sns.countplot('DayOfWeek', data=train, ax=ax[0])
ax[0].set_title('Nb of flights by day of week')
sns.countplot('DayOfWeek', hue='delayed', data=train, ax=ax[1])
ax[1].set_title('Delayed or not flights by day of week')
sns.barplot('DayOfWeek', 'delayed', data=train, ax=ax[2])
ax[2].set_title('Rate of delayed flights by day of week')


# * Here we can see that Thursday and Friday have the highest rates of delayed flights where as Tuesday, Wednesday and Saturday have the lowest

# ## 4- Departure Time

# In[ ]:


plt.hist(train.DepTime)
plt.xlabel('Departure Time')


# * We will come back to this variable once we bin it because of the large spectrum of values

# ## 5- Unique Carrier

# In[ ]:


# Nb of unique values
len(set(train.UniqueCarrier))


# In[ ]:


fig, ax = plt.subplots(3, 1, figsize=(15,15))
sns.countplot('UniqueCarrier', data=train, ax=ax[0])
ax[0].set_title('Nb of flights per unique carrier')
sns.countplot('UniqueCarrier', hue='delayed', data=train, ax=ax[1])
ax[1].set_title('Nb of delayed/not flights by unique carrier')
sns.barplot('UniqueCarrier', 'delayed', data=train, ax=ax[2])
ax[2].set_title('Rate of delayed flights by unique carrier')


# * We can see that the Unique carrier variable can have a good role in the delays

# ## 6- Origin/Destination

# In[ ]:


# Nb of unique values
print(len(set(train.Origin)))
print(len(set(train.Dest)))


# * Too many categorical values to plot. Maybe it would be a good idea to create a variable with routes Origin - Destination

# ## 7- Distance

# In[ ]:


plt.hist(train.Distance)
plt.xlabel('Distance')


# * We can see that most of the flights are short in distance and less than 1000 miles
# * Would it be a good idea to normalize and/or scale this variable or is the difference more meaningful this way?
# * Maybe bin this variable?

# # Feature Engineering

# ## 1- New features

# In[ ]:


all_data['Route'] = all_data['Origin'] + all_data['Dest']


# In[ ]:


all_data['UniqueCarrier_Origin'] = all_data['UniqueCarrier'] + "_" + all_data['Origin']
all_data['UniqueCarrier_Dest'] = all_data['UniqueCarrier'] + "_" + all_data['Dest']


# In[ ]:


all_data['is_weekend'] = (all_data['DayOfWeek'] == 6) | (all_data['DayOfWeek'] == 7)


# In[ ]:


# Hour and minute
all_data['hour'] = all_data['DepTime'] // 100
all_data.loc[all_data['hour'] == 24, 'hour'] = 0
all_data.loc[all_data['hour'] == 25, 'hour'] = 1
all_data['minute'] = all_data['DepTime'] % 100


# In[ ]:


# give more importance to hour variable
all_data['hour_sq'] = all_data['hour'] ** 2
all_data['hour_sq2'] = all_data['hour'] ** 4


# ## 2- Binning

# #### Season

# In[ ]:


all_data['summer'] = (all_data['Month'].isin([6, 7, 8]))
all_data['autumn'] = (all_data['Month'].isin([9, 10, 11]))
all_data['winter'] = (all_data['Month'].isin([12, 1, 2]))
all_data['spring'] = (all_data['Month'].isin([3, 4, 5]))


# #### Departure Time

# In[ ]:


all_data['DayTime'] = 0
all_data.loc[all_data.DepTime <= 600 , 'DepTime_bin'] = 'Night'
all_data.loc[(all_data.DepTime > 600) & (all_data.DepTime <= 1200), 'DepTime_bin'] = 'Morning'
all_data.loc[(all_data.DepTime > 1200) & (all_data.DepTime <= 1800), 'DepTime_bin'] = 'Afternoon'
all_data.loc[(all_data.DepTime > 1800) & (all_data.DepTime <= 2600), 'DepTime_bin'] = 'Evening'


# In[ ]:


all_data['DepTime_bin'] = 0
all_data.loc[all_data.DepTime <= 600 , 'DepTime_bin'] = 'vem'
all_data.loc[(all_data.DepTime > 600) & (all_data.DepTime <= 900), 'DepTime_bin'] = 'm'
all_data.loc[(all_data.DepTime > 900) & (all_data.DepTime <= 1200), 'DepTime_bin'] = 'mm'
all_data.loc[(all_data.DepTime > 1200) & (all_data.DepTime <= 1500), 'DepTime_bin'] = 'maf'
all_data.loc[(all_data.DepTime > 1500) & (all_data.DepTime <= 1800), 'DepTime_bin'] = 'af'
all_data.loc[(all_data.DepTime > 1800) & (all_data.DepTime <= 2100), 'DepTime_bin'] = 'n'
all_data.loc[(all_data.DepTime > 2100) & (all_data.DepTime <= 2400), 'DepTime_bin'] = 'nn'
all_data.loc[all_data.DepTime > 2400, 'DepTime_bin'] = 'lm'
all_data = all_data.drop(['DepTime'], axis=1)


# #### Distance

# In[ ]:


all_data['Dist_bin'] = 0
all_data.loc[all_data.Distance <= 500 , 'Dist_bin'] = 'vshort'
all_data.loc[(all_data.Distance > 500) & (all_data.Distance <= 1000), 'Dist_bin'] = 'short'
all_data.loc[(all_data.Distance > 1000) & (all_data.Distance <= 1500), 'Dist_bin'] = 'mid'
all_data.loc[(all_data.Distance > 1500) & (all_data.Distance <= 2000), 'Dist_bin'] = 'midlong'
all_data.loc[(all_data.Distance > 2000) & (all_data.Distance <= 2500), 'Dist_bin'] = 'long'
all_data.loc[all_data.Distance > 2500, 'Dist_bin'] = 'vlong'
all_data = all_data.drop(['Distance'], axis=1)


# # Predictive Modeling

# In[ ]:


new_train = all_data.iloc[:100000]
new_test = all_data.iloc[100000:]


# In[ ]:


feature_columns = list(new_train.columns)
feature_columns.remove('delayed')


# In[ ]:


X = new_train[feature_columns]
y = new_train.delayed

#split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size= 0.2, random_state=1)


# ## Catboost

# In[ ]:


all_data.head()


# In[ ]:


feature_columns


# In[ ]:


model_ctb = CatBoostClassifier(iterations=3000, loss_function='Logloss',
                               l2_leaf_reg=0.8, od_type='Iter',
                               random_seed=17, silent=True)
#model_ctb = GridSearchCV(model_ctb, {'learning_rate':[0.5, 0.1], 'n_estimators':[500, 1000]})
model_ctb.fit(X_train, y_train.astype(int), cat_features=[2, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20])
predictions = model_ctb.predict_proba(X_val)[:, 1]
accuracy = roc_auc_score(y_val.astype(int), predictions)
print('Accuracy Catboost: ', accuracy)


# # Results

# In[ ]:


model_ctb.fit(X, y.astype(int), cat_features=[2, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20])


# In[ ]:


sample = pd.read_csv("../input/flight-delays-fall-2018/sample_submission.csv.zip", compression='zip')
sample.head()


# In[ ]:


predictions = model_ctb.predict_proba(new_test[feature_columns])[:, 1]


# In[ ]:


submission = pd.DataFrame({'id':range(100000),'dep_delayed_15min':predictions})
submission.head(900)


# In[ ]:


filename = 'flight_delay.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)

