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


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:





# In[ ]:


heart_df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')


# In[ ]:


heart_df.head()


# In[ ]:


heart_df.info()


# In[ ]:


heart_df.isna().sum()


# In[ ]:


heart_df['target'].value_counts()


# In[ ]:


sns.countplot(heart_df['target'])


# In[ ]:


cat_feats = [col for col in heart_df.columns if heart_df[col].nunique() <10]
cont_feats = [col for col in heart_df.columns if col not in cat_feats]


# In[ ]:


print('Categorical features', end=": "),
print(cat_feats)
print('Continuous features',end=": "),
print(cont_feats)


# In[ ]:


for col in cat_feats:
    print(heart_df[col].value_counts())


# There are no null values

# In[ ]:


plt.figure(figsize=(15, 15))

for i, column in enumerate(cont_feats, 1):
    plt.subplot(3, 2, i)
    heart_df[heart_df["target"] == 0][column].hist(bins=35, color='blue', label='Heart Disease = NO', alpha=0.6)
    heart_df[heart_df["target"] == 1][column].hist(bins=35, color='green', label='Heart Disease = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[ ]:


plt.figure(figsize=(15, 15))

for i, column in enumerate(cat_feats, 1):
    plt.subplot(3, 3, i)
    heart_df[heart_df["target"] == 0][column].hist(bins=35, color='blue', label='Heart Disease = NO', alpha=0.6)
    heart_df[heart_df["target"] == 1][column].hist(bins=35, color='green', label='Heart Disease = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[ ]:


df = heart_df
for col in cat_feats:
    df[col] = df[col].apply(str)
#check
df.info()


# In[ ]:


# Converting categorical features into dummies
df = pd.get_dummies(df, columns = cat_feats)


# In[ ]:


#Using standard scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

df[cont_feats] = sc.fit_transform(df[cont_feats])


# In[ ]:


df.head()


# In[ ]:


X = df.drop(['target_0','target_1'], axis = 1)
y = heart_df['target']


# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:


len(X.columns)


# In[ ]:


# Deep learning
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Create a model with one hidden layer
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
#input layer consists of 29 inputs
model.add(Dense(32, activation='relu', input_shape=(30,))) 
model.add(Dense(16, activation='relu'))

# Add an output layer with one output and sigmoid activation
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                   
model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)
y_nn_pred = model.predict(X_test)


# In[ ]:


y_train


# In[ ]:


print(confusion_matrix(y_test, y_nn_pred.round()))
print(classification_report(y_test, y_nn_pred.round()))


# In[ ]:


submission = pd.DataFrame({'Id':y_test.index,'predict':y_pred})
submission.to_csv('submission.csv')


# In[ ]:




