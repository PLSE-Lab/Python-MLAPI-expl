#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Load the Data

# In[ ]:


train_df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
train_df.head(5)


# # Checking the Missing values

# In[ ]:


train_df.isnull().sum()


# In[ ]:


train_df.isna().sum()


# # Data Visualization
# let's try some data visualization techniques here, first lets analyze the correlation among the values

# In[ ]:


corr = train_df.corr()
corr.values


# In[ ]:


plt.figure(figsize=(17,7))
sns.heatmap(corr)


# correlation between Blood pressure, Glucose and outcome

# In[ ]:


plt.figure(figsize=(17,7))
plt.subplot(2,2,1)
sns.boxplot(train_df['Outcome'],train_df['BloodPressure'])
plt.subplot(2,2,2)
sns.boxplot(train_df['Outcome'],train_df['Glucose'])


# In[ ]:


plt.figure(figsize=(17,7))
sns.boxplot(train_df['Outcome'],train_df['Age'])


# # Model development

# In[ ]:


X = train_df.drop(['Outcome'], axis =1).values
y = train_df['Outcome'].values


# In[ ]:


X[0:5]


# In[ ]:


y[0:5]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=40)
print('training data : ',X_train.shape, y_train.shape)
print('testing data : ',X_test.shape, y_test.shape)


# # Loading the Decision Tree model

# In[ ]:


dtc = DecisionTreeClassifier()
dtc = dtc.fit(X_train, y_train)
dtc


# Prediction part

# In[ ]:


y_pred = dtc.predict(X_test)


# Lets calcluate the accuracy score!!

# In[ ]:


acc = accuracy_score(y_pred, y_test)
print('Accuracy is : ',acc)


# The simple DecisionTreeClassifier is demonstrated do upvote it if you liked it and you can edit it for modifications that you think is necessary and do comment if any changes can be made.
# Cheers!!

# # PCA

# In[ ]:


from sklearn.decomposition import PCA

pca = PCA(3)
X_pca = pca.fit_transform(X_train)


# In[ ]:


print(pca.explained_variance_ratio_)


# # Random Forrest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)


# In[ ]:


y_preds = rfc.predict(X_test)
acc_rfc = accuracy_score(y_preds,y_test)
acc_rfc


# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)


# In[ ]:


X_transform = scaler.transform(X)


# # Neural networks approach

# In[ ]:


classifier =Sequential()

# create input layer
classifier.add(Dense(units=6,kernel_initializer='uniform' , activation='tanh' , input_dim=8))

# create hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform' , activation='tanh'))
# create hidden layer

# create output layer
classifier.add(Dense(units=1 , kernel_initializer='uniform' , activation='sigmoid'))

#compiling the ANN
classifier.compile(optimizer='adam' , loss='binary_crossentropy' , metrics=['accuracy'])


# In[ ]:


classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


print(accuracy_score(y_preds,y_test))


# In[ ]:




