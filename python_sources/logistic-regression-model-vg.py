#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Read Training Data**

# In[ ]:


import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# In[357]:


train_data = pd.read_csv("../input/train.csv")
train_data.head(5)


# In[358]:


test_data = pd.read_csv("../input/test.csv")
test_data.head(5)


# In[359]:


train_data.describe()


# In[360]:


test_data.describe()


# In[361]:


train_data.info()


# In[362]:


test_data.info()


# Change variable types

# In[363]:


train_data = train_data.astype({"Survived":'category', "Pclass":'category', "Sex":'category', "SibSp":'category', "Parch":'category', "Embarked":'category'})

test_data = test_data.astype({"Pclass":'category', "Sex":'category', "SibSp":'category', "Parch":'category', "Embarked":'category'})


# In[364]:


train_data.isna().sum()


# In[365]:


test_data.isna().sum()


# **Impute missing values for Fare, Age**

# In[366]:


test_data['Fare'] = test_data['Fare'].fillna(0)


# In[367]:


#train_data.Age.mean()
#test_data.Age.mean()
train_data.Embarked.mode()


# In[368]:


train_data['Age'] = train_data['Age'].fillna(29.69)


# In[369]:


test_data['Age'] = test_data['Age'].fillna(30.27)


# In[370]:


train_data['Embarked'] = train_data['Embarked'].fillna("S")


# In[371]:


train_data.isnull().sum()


# In[268]:


#train_data = train_data.dropna()
#test_data = test_data.dropna()


# **Segregate dependent and independent vars**

# In[372]:


y_train = train_data["Survived"]
#X_train = train_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
X_train = train_data[["Pclass", "Sex", "SibSp", "Fare", "Age", "Parch", "Embarked"]]


# In[373]:


#y_test = test_data["Survived"]
#X_test = test_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
X_test = test_data[["Pclass", "Sex", "SibSp", "Fare", "Age", "Parch", "Embarked"]]


# In[374]:


test_data.shape


# In[375]:


X_train_d = pd.get_dummies(X_train, columns=["Pclass", "Sex", "SibSp", "Parch", "Embarked"])


# In[376]:


X_test_d = pd.get_dummies(X_test, columns=["Pclass", "Sex", "SibSp", "Parch", "Embarked"])


# In[377]:


X_train_d.head()


# In[391]:


X_test_d.head()


# In[398]:


X_test_d = X_test_d.drop(['Parch_9'], axis=1)


# In[399]:


print(X_train_d.shape)
print(X_test_d.shape)


# In[400]:


scaler = StandardScaler().fit(X_train_d)
X_train_scale = scaler.transform(X_train_d)
X_train_scale = pd.DataFrame(X_train_scale)
X_train_scale.head()


# In[401]:


X_test_d.head()


# In[402]:


scaler1 = StandardScaler().fit(X_test_d)
X_test_scale = scaler1.transform(X_test_d)
X_test_scale = pd.DataFrame(X_test_scale)
X_test_scale.head()


# In[403]:


sns.heatmap(X_train_d.corr())


# ** Encode categorical variables**

# In[404]:


y_train.head()


# In[405]:


model = LogisticRegression()
model.fit(X_train_d, y_train)


# Scaler Model

# In[406]:


model_s = LogisticRegression()
model_s.fit(X_train_scale, y_train)


# In[407]:


model.coef_


# In[408]:


model_s.coef_


# In[409]:


print(X_test_d.shape)
print(y_pred_df.shape)


# In[410]:


y_pred = model.predict(X_test_d)
y_pred_df = pd.DataFrame(y_pred, columns=["Survived"])
print(y_pred_df.head())


# In[411]:


y_pred_scale = model.predict(X_test_scale)
y_pred_df_scale = pd.DataFrame(y_pred_scale, columns=["Survived"])
print(y_pred_df_scale.head())


# In[412]:


print(y_pred_df.shape)
print(y_pred_df_scale.shape)


# In[413]:


pred_df = pd.concat([test_data, y_pred_df], axis=1, sort=False)

pred_df_scale = pd.concat([test_data, y_pred_df_scale], axis=1, sort=False)


# In[414]:


print(pred_df.shape)
print(pred_df_scale.shape)


# In[415]:


pred_df.head()


# In[416]:


pred_df_scale.head()


# In[417]:


predictions = pred_df[["PassengerId", "Survived"]]
predictions.head()


# In[418]:


predictions_s = pred_df_scale[["PassengerId", "Survived"]]
predictions_s.head()


# In[419]:


predictions_s.to_csv("predictions_s_4.csv")

