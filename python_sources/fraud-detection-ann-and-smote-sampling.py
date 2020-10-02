#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection - Neural Networks and SMOTE Sampling

# ## About Dataset
# <br>
# "The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# 
# It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise."
# <br>
# <img src="https://images.techhive.com/images/article/2014/12/credit_card_fraud-100537848-large.jpg"/>

# ## Import Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Read and Explore Data

# In[ ]:


df = pd.read_csv("../input/creditcard.csv")


# In[ ]:


# First 5 rows of data
df.head()


# In[ ]:


df.info()


# In[ ]:


df.columns


# ## Normalize 'Amount'

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


df['Amount(Normalized)'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))


# In[ ]:


df.iloc[:,[29,31]].head()


# In[ ]:


df = df.drop(columns = ['Amount', 'Time'], axis=1) # This columns are not necessary anymore.


# ## Data PreProcessing

# In[ ]:


X = df.drop('Class', axis=1)

y = df['Class']


# ## Train-Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


# We are transforming data to numpy array to implementing with keras
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# ## Artificial Neural Networks

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout 


# In[ ]:


model = Sequential([
    Dense(units=20, input_dim = X_train.shape[1], activation='relu'),
    Dense(units=24,activation='relu'),
    Dropout(0.5),
    Dense(units=20,activation='relu'),
    Dense(units=24,activation='relu'),
    Dense(1, activation='sigmoid')
])


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=30, epochs=5)


# In[ ]:


score = model.evaluate(X_test, y_test)
print('Test Accuracy: {:.2f}%\nTest Loss: {}'.format(score[1]*100,score[0]))


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


y_pred = model.predict(X_test)
y_test = pd.DataFrame(y_test)


# In[ ]:


cm = confusion_matrix(y_test, y_pred.round())


# In[ ]:


sns.heatmap(cm, annot=True, fmt='.0f', cmap='cividis_r')
plt.show()


# Our results is fine however it is not the best way to do things like that. Since our dataset is unbalanced (we have 492 frauds out of 284,807 transactions) we will use 'smote sampling'. Basically smote turn our inbalanced data to balanced data.
# For brief explanation you can check the link: http://rikunert.com/SMOTE_explained

# ## SMOTE Sampling

# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:


X_smote, y_smote = SMOTE().fit_sample(X, y)


# In[ ]:


X_smote = pd.DataFrame(X_smote)
y_smote = pd.DataFrame(y_smote)


# In[ ]:


y_smote.iloc[:,0].value_counts()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.3, random_state=0)


# In[ ]:


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size = 30, epochs = 5)


# In[ ]:


score = model.evaluate(X_test, y_test)
print('Test Accuracy: {:.2f}%\nTest Loss: {}'.format(score[1]*100,score[0]))


# In[ ]:


y_pred = model.predict(X_test)
y_test = pd.DataFrame(y_test)
cm = confusion_matrix(y_test, y_pred.round())
sns.heatmap(cm, annot=True, fmt='.0f')
plt.show()


# It is not the true result 'cause we used data with smote sampling because of that number of class 0 and class 1 are equal in here. That's why we'll use whole data we imported at the beginning.

# In[ ]:


y_pred2 = model.predict(X)
y_test2 = pd.DataFrame(y)
cm2 = confusion_matrix(y_test2, y_pred2.round())
sns.heatmap(cm2, annot=True, fmt='.0f', cmap='coolwarm')
plt.show()


# In[ ]:


scoreNew = model.evaluate(X, y)
print('Test Accuracy: {:.2f}%\nTest Loss: {}'.format(scoreNew[1]*100,scoreNew[0]))


# In[ ]:


print(classification_report(y_test2, y_pred2.round()))


# **Thank you, if you like it please upvote and make a comment.**
