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


import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


data = pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')
data.head()


# In[ ]:


bool_data = data.isnull()
for feature in list(data.columns):
    print(feature + 'null: ' + str(bool_data[feature].sum()))
#There are no null values in the dataset.


# In[ ]:


corr_matrix = data.corr()
sns.heatmap(corr_matrix, cmap='YlGnBu')


# In[ ]:


corr_features = dict(corr_matrix['price_range'])
least_corr_features = []
for feature in corr_features:
    if abs(corr_features[feature]) < 0.01:
        least_corr_features.append(feature)
        
print(least_corr_features)  #features that have very less influence on the outcome.


# In[ ]:


#Its a classification problem.
#check for the labels in the price_range.
data['price_range'].value_counts()
#there are four price ranges. The data is balanced as all the classes have equal count of examples.


# # **Lets analyse two best features**
# 1) ram_size
# 
# 2) battery_power

# In[ ]:


ram_mean = dict(data.groupby('price_range').mean().ram)
keys = ram_mean.keys()
values = ram_mean.values()
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
ax.bar(keys, values)
ax.legend(['price_range', 'ram_size'])
ax.set_xlabel('price_range')
ax.set_ylabel('ram_size')
#Its obvious that as ram size increases the cost of the mobile also increases. RAM is costly!!


# In[ ]:


battery_power = dict(data.groupby('price_range').mean().battery_power)
keys = battery_power.keys()
values = battery_power.values()
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
ax.bar(keys, values)
ax.legend(['price_range', 'battery_power'])
ax.set_xlabel('price_range')
ax.set_ylabel('battery_power')


# In[ ]:


clock_speed = dict(data.groupby('price_range').mean().clock_speed)
n_cores = dict(data.groupby('price_range').mean().n_cores)
print('We cannot draw some important conclusions with the help of these features.')
print(clock_speed)
print(n_cores)


# # **Bluetooth vs Price Range**

# In[ ]:


#bluetooth vs price_range.
sns.countplot('price_range', hue = 'blue', data = data)


# # **Mobile Weight vs Price Range**

# In[ ]:


#Usually high cost phones weigh less.
sns.boxplot('price_range', 'mobile_wt', data = data)


# # **First Try: Considering the low performing features**

# In[ ]:


labels = data['price_range']
train_data = data.drop('price_range', axis = 1)
print(train_data.shape, labels.shape)

X_train, X_valid, Y_train, Y_valid = train_test_split(train_data, labels, test_size = 0.2)
print(X_train.shape, X_valid.shape)

scaler = MinMaxScaler(feature_range=(0,4))
X_train = scaler.fit_transform(X_train)
X_valid = scaler.fit_transform(X_valid)

classifier = LogisticRegression()
classifier.fit(X_train, Y_train)

Y_preds = classifier.predict(X_valid)
print(f"Test score: {classifier.score(X_valid, Y_valid)}")


# In[ ]:


#confusion matrix for the first try.
print(classification_report(Y_valid,Y_preds))


# # **Second Try: Dropping the low performing features**

# In[ ]:


#Train without using the weak features.
labels2 = data['price_range']
train_data2 = data.drop(least_corr_features + ['price_range'], axis = 1)
print(train_data2.shape, labels2.shape)

X_train2, X_valid2, Y_train2, Y_valid2 = train_test_split(train_data2, labels2, test_size = 0.2)
print(X_train2.shape, X_valid2.shape)

scaler = MinMaxScaler(feature_range=(0,4))
X_train2 = scaler.fit_transform(X_train2)
X_valid2 = scaler.fit_transform(X_valid2)

classifier = LogisticRegression()
classifier.fit(X_train2, Y_train2)

Y_preds2 = classifier.predict(X_valid2)
print(f"Test score: {classifier.score(X_valid2, Y_valid2)}")
#Model could not perform better than the one trained previously. Hence its better to use the dropped features.
#Even though their contribution is very less the model could learn something from them!!
#Dropping them would be useful when we have many features to look on. But here we just have 20 features.


# In[ ]:


#confusion matrix for the second model.
print(classification_report(Y_valid2,Y_preds2))


# # **Third try: Random Forest**

# In[ ]:


#need not scale the data for random forest. Tune in the number of trees and try getting better test score.
labels = data['price_range']
train_data = data.drop(least_corr_features + ['price_range'], axis = 1)
print(train_data.shape, labels.shape)

X_train, X_valid, Y_train, Y_valid = train_test_split(train_data, labels, test_size = 0.3)
print(X_train.shape, X_valid.shape)

rfc = RandomForestClassifier(n_estimators=400)
rfc.fit(X_train, Y_train)

rfc.score(X_valid,Y_valid)


# # **Please upvote if you find this helpful**

# In[ ]:




