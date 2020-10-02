#!/usr/bin/env python
# coding: utf-8

# # Bank-Marketing-Prediction

# 

# In[ ]:


# import all necessary libraries
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

#
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# model metrics 
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

# import models

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier

import os

import warnings
warnings.filterwarnings('ignore')


# # Get the data

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# load the data
df = pd.read_csv('/kaggle/input/bank-marketing-dataset/bank.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


# get a description of our features
df.info()


# In[ ]:


df.describe()


# # Explore and visualize the data

# In[ ]:


# create an additional dataframe that holds features upon exploration
df_features = pd.DataFrame()


# In[ ]:


# check the datatypes of our features
df.dtypes


# In[ ]:


# describe our numerical data
df.describe()


# In[ ]:


# describe our categorical data
df.describe(include=['O'])


# In[ ]:


df.head()


# **Target variable: y**

# In[ ]:


# used to convert columns with values (no, yes) into numerical values of (0, 1)
def yes_no_encoder(data):
    if 'no' in data:
        data = 0
    elif 'yes' in data:
        data = 1
    
    return data


# In[ ]:


# convert our categorical target to numeric
df['deposit'] = df['deposit'].apply(yes_no_encoder)


# In[ ]:


df_features['deposit'] = df['deposit']


# In[ ]:


fig = plt.figure(figsize=(20, 1))
sns.countplot(y='deposit', data=df)
print(df.deposit.value_counts())


# **Feature: Age**

# In[ ]:


df_features['age'] = df['age']


# In[ ]:


sns.distplot(df_features['age'], kde=False)


# **Feature: Job**

# In[ ]:


df_features['job'] = df['job']


# In[ ]:


plt.figure(figsize=(15, 5))
sns.countplot(y='job', data=df_features)


# **Feature: marital**

# In[ ]:


df_features['marital'] = df['marital']


# In[ ]:


# returns percentage distribution of all categorical items in a specified column
def value_perc(feature):
    perc = feature.value_counts(normalize=True).reset_index()
    perc.columns = ['value', 'perc']
    perc['perc'] = round( perc['perc'] * 100 , 2)
    return perc


# In[ ]:


sns.countplot(y='marital', data=df_features)
print(value_perc(df_features.marital))


# **Feature: Education**

# In[ ]:


df_features['education'] = df['education']


# In[ ]:


sns.countplot(y='education', data=df_features)


# **Feature: default**

# In[ ]:


df_features['default'] = df['default']


# In[ ]:


# conver categorical default values to numeric values
df_features['default'] = df_features['default'].apply(yes_no_encoder)


# In[ ]:


sns.countplot(y='default', data=df_features)
print(value_perc(df['default']))


# **Feature: Balance**

# In[ ]:


df_features['balance'] = df['balance']


# In[ ]:


sns.distplot(df_features['balance'])
print("The mean balance: ", round(df_features['balance'].mean(), 2))
print("The mean balance: ", round(df_features['balance'].std(), 2))


# In[ ]:


df.head()


# **Feature: Housing**

# In[ ]:


df_features['housing'] = df['housing']


# In[ ]:


df_features['housing'] = df_features['housing'].apply(yes_no_encoder) # convert yes/no to numeric equivalent


# In[ ]:


sns.countplot(y='housing', data=df)
print(value_perc(df['housing']))


# **Feature: Loan**

# In[ ]:


df_features['loan'] = df['loan']


# In[ ]:


df_features['loan'] = df_features['loan'].apply(yes_no_encoder)


# In[ ]:


sns.countplot(y='loan', data=df)


# **Feature: Contract**

# In[ ]:


df_features['contact'] = df['contact']


# In[ ]:


sns.countplot(y='contact', data=df_features)


# **Feature: Day**

# In[ ]:


df_features['day'] = df['day']


# In[ ]:


sns.distplot(df_features['day'], kde=False)
print("Mean number of day: ", df_features['day'].mean())
print("Mean number of day: ", df_features['day'].std())


# **Feature: month**

# In[ ]:


df_features['month'] = df['month']


# In[ ]:


sns.countplot(y='month', data=df_features)
print(value_perc(df_features['month']))


# In[ ]:





# **Feature: Duration**

# In[ ]:


df_features['duration'] = df['duration']


# In[ ]:


sns.distplot(df_features['duration'])
print("Mean duration: ", df_features['duration'].mean())
print("Std.Dev duration: ", df_features['duration'].std())


# **Feature: Campaign**

# In[ ]:


df_features['campaign'] = df['campaign']


# In[ ]:


sns.distplot(df_features['campaign'])


# **Feature: Pdays**

# In[ ]:


df_features['pdays'] = df['pdays']


# In[ ]:


sns.distplot(df_features['pdays'])


# **Feature: Previous**

# In[ ]:


df_features['previous'] = df['previous']


# In[ ]:


sns.distplot(df_features['previous'])


# **Feature: Poutcome**

# In[ ]:


df_features['poutcome']=  df['poutcome']


# In[ ]:


sns.countplot(y='poutcome', data=df_features)


# In[ ]:


# check for missing values
def missing_values(data):
    return data.isnull().sum()

missing_values(df_features)


# # Prepare the data

# **Feature encoding**

# In[ ]:


# get all categorical feature
obj_cols = df_features.select_dtypes(include=['object']).columns


# In[ ]:


# get dummies for string features
df_features = pd.get_dummies(df_features, columns=obj_cols, drop_first=True)


# **Train/test split**

# In[ ]:


y = df_features['deposit']
X = df_features.drop(['deposit'], axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_cols = X_train.columns


# **Feature Scaling**

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # Building the machine learning model

# In[ ]:


# train the model and use it to predict the label for unseen data
def fit_ml_algo(algo, X_train, y_train, X_test, y_test):
    
    model = algo.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = round(accuracy_score(y_pred, y_test) * 100, 2)
    cf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = precision_score(y_pred, y_test)
    f1 = f1_score(y_test, y_pred)
     
    return acc, cf_matrix, precision, recall, f1, model


# **Logistic regression**

# In[ ]:


acc, cf_matrix, precision, recall, f1, model = fit_ml_algo(LogisticRegression(), X_train, y_train, X_test, y_test)
ax = sns.heatmap(cf_matrix, annot=True, fmt='g') #notation: "annot" not "annote"
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
print("Accuracy: ", acc)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)


# **Random Forest**

# In[ ]:


acc, cf_matrix, precision, recall, f1, model = fit_ml_algo(RandomForestClassifier(), X_train, y_train, X_test, y_test)
ax = sns.heatmap(cf_matrix, annot=True, fmt='g') #notation: "annot" not "annote"
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
print("Accuracy: ", acc)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)


# **Support Vector Machine**

# In[ ]:


acc, cf_matrix, precision, recall, f1, model = fit_ml_algo(LinearSVC(), X_train, y_train, X_test, y_test)
ax = sns.heatmap(cf_matrix, annot=True, fmt='g') #notation: "annot" not "annote"
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
print("Accuracy: ", acc)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)


# **Gradient Boosting**

# In[ ]:


acc, cf_matrix, precision, recall, f1, model = fit_ml_algo(GradientBoostingClassifier(), X_train, y_train, X_test, y_test)
ax = sns.heatmap(cf_matrix, annot=True, fmt='g') #notation: "annot" not "annote"
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
print("Accuracy: ", acc)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)


# **Adaboost**

# In[ ]:


acc, cf_matrix, precision, recall, f1, model = fit_ml_algo(AdaBoostClassifier(n_estimators=100), X_train, y_train, X_test, y_test)
ax = sns.heatmap(cf_matrix, annot=True, fmt='g') #notation: "annot" not "annote"
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
print("Accuracy: ", acc)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)


# In[ ]:


pd.DataFrame({'feature': X_cols, 'Importance': model.feature_importances_})


# In[ ]:




