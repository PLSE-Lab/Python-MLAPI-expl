#!/usr/bin/env python
# coding: utf-8

# In[75]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[76]:


# we set a random seed so something random won't affect our results
np.random.seed(2)


# In[77]:


data = pd.read_csv('../input/creditcard.csv')


# * **EXPLORING AND CLEANING DATA**

# In[78]:


data.head()


# As it can be seen our data columns do not have any name so we can't see what numbers really mean here just by looking at them.
# The amount column isn't normalized, so we will normalize it to be in same range as other values

# In[79]:


data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Amount', 'Time'], axis=1)


# In[80]:


data.head()


# In[81]:


sns.countplot(data['Class'])


# as can be viewed above our dataset is extremely unbalanced, with almost all of our data belonging to non-fraudulent 
# transactions.

# In[82]:


X = data.iloc[:, data.columns != 'Class']
Y = data.iloc[:, data.columns == 'Class']


# In[83]:


X.corrwith(data.Class).plot.bar(figsize=(20, 10), fontsize=12, grid=True)


# In[84]:


plt.figure(figsize=(20, 10))
sns.heatmap(data.corr(), annot= True)


# the plot above shows the relationship between the features in our dataset. as it can be seen almost all of our features are independant of each other.

# In[85]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


# In[86]:


X_train.shape


# * **Classification with Neural Network**

# In[87]:


model = Sequential()
model.add(Dense(units=16, input_dim= 29, activation='relu'))
model.add(Dense(units=24, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=20, activation='relu'))
model.add(Dense(units=24, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()


# In[88]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[89]:


model.fit(X_train, Y_train, batch_size=15, epochs=5, validation_split=0.2)


# In[90]:


model.evaluate(X_test, Y_test)


# In[91]:


y_pred = model.predict(X_test)


# In[92]:


cm_matrix = confusion_matrix(Y_test, y_pred.round())


# In[93]:


print(cm_matrix)


# While our test results show that we have 99.9% percent accuracy, around 25% of all frauds where undetected<br>
# and almost all non-fraudulent cases where predicted correctly. this is because of **extreme imbalance in our dataset**<br>
# for ratio of two classes compared to each other.

# * **Random Forest**

# In[94]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train.values.ravel())


# In[95]:


y_pred = random_forest.predict(X_test)


# In[96]:


random_forest.score(X_test, Y_test)


# In[97]:


cm_matrix = confusion_matrix(Y_test, y_pred.round())


# In[98]:


print(cm_matrix)


# * **Improving the ratio between classes**
# 1. Undersampling

# In[99]:


fraud_indices = np.array(data[data.Class == 1].index)
non_frud_indices = data[data.Class == 0].index
num_fraud = len(fraud_indices)
print(num_fraud)


# In[100]:


random_normal = np.array(np.random.choice(non_frud_indices, num_fraud, replace=False))
print(len(random_normal))


# In[101]:


undersample_idx = np.concatenate([fraud_indices, random_normal])


# In[102]:


new_data = data.iloc[undersample_idx, :]


# In[103]:


X_under = new_data.iloc[:, new_data.columns != 'Class']
Y_under = new_data.iloc[:, new_data.columns == 'Class']

X_train, X_test, Y_train, Y_test = train_test_split(X_under, Y_under, test_size=0.3, random_state=0)


# In[104]:


# due to the small amount of data we use random forests instead of neural network and decrease number of estimators to improve performance

random_forest = RandomForestClassifier(n_estimators=10)

random_forest.fit(X_train, Y_train.values.ravel())
y_pred = random_forest.predict(X_test)
print(random_forest.score(X_test, Y_test))

cm_matrix = confusion_matrix(Y_test, y_pred.round())
print(cm_matrix)


# it is visible that our accuracy for detection of fraud cases has been improved with underSampling

# * **OverSampling **<br>
# **SMOTE**

# In[105]:


X_over, Y_over = SMOTE().fit_sample(X, Y.values.ravel())


# In[106]:


X_train, X_test, Y_train, Y_test = train_test_split(X_over, Y_over, test_size=0.3, random_state=0)


# In[107]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=15, epochs=5, validation_split=0.2)


# In[108]:


model.evaluate(X_test, Y_test)
y_pred = model.predict(X_test)

cm_matrix = confusion_matrix(Y_test, y_pred.round())
print(cm_matrix)

