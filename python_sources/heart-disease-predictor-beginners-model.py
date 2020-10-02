#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df.head()


# ABOUT THE FEATURES : age: The person's age in years
# 
# sex: The person's sex (1 = male, 0 = female)
# 
# cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
# 
# trestbps: The person's resting blood pressure (mm Hg on admission to the hospital) chol: The person's cholesterol measurement in mg/dl
# 
# fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
# 
# restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
# 
# thalach: The person's maximum heart rate achieved
# 
# exang: Exercise induced angina (1 = yes; 0 = no)
# 
# oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)
# 
# slope: the slope of the peak exercise ST segment (****Value 1: upsloping, Value 2: flat, Value 3: downsloping)
# 
# ca: The number of major vessels (0-3)
# 
# thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect) target: Heart disease (0 = no, 1 = yes)

# In[ ]:


df['target'].value_counts()


# In[ ]:


sns.countplot(df['target'])


# In[ ]:


sns.distplot(df['age'])


# In[ ]:


df.groupby('target')['age'].mean()


# In[ ]:


fg = sns.FacetGrid(col = 'target',data=df)
fg.map(plt.hist,'age')


# In[ ]:


df.groupby('target')['sex'].value_counts()


# In[ ]:


sns.countplot(df['target'],hue =df['sex'])


# In[ ]:


plt.figure(figsize = (10,5))
sns.heatmap(df.corr(),annot=True,linewidths=1)


# Since Thalach and cp are the only features mildly correlated with the Target Variable

# In[ ]:


sns.kdeplot(df['thalach'],shade = True)


# In[ ]:


sns.stripplot(x=df['target'],y=df['thalach'])


# In[ ]:


sns.boxplot(x=df['target'],y=df['thalach'])


# In[ ]:


df.groupby('target')['cp'].value_counts()


# In[ ]:


sns.countplot(df['target'],hue = df['cp'])


# FEATURE ENGINEERING

# In[ ]:


df.shape


# In[ ]:


df.nunique()


# Since 'cp', 'thal' and 'slope' are categorical variables we'll turn them into dummy variables

# In[ ]:


cp_new = pd.get_dummies(df['cp'], prefix = "cp")
thal_new = pd.get_dummies(df['thal'], prefix = "thal")
slope_new = pd.get_dummies(df['slope'], prefix = "slope")


# In[ ]:


frames = [df, cp_new, thal_new, slope_new]
df = pd.concat(frames, axis = 1)
df.head()


# In[ ]:


df = df.drop(['cp', 'thal', 'slope'],axis=1)
df.head()


# **Creating Models**

# In[ ]:


X = df.drop(['target'],axis=1)
y = df['target']


# In[ ]:


# Normalize
X = (X - np.min(X)) / (np.max(X) - np.min(X)).values


# In[ ]:


from sklearn.model_selection import train_test_split
X = df.drop(['target'],axis=1)
y = df['target']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


# Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))


# In[ ]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))


# In[ ]:


from sklearn.metrics import plot_roc_curve
plot_roc_curve(lr,X_test,y_test)


# K Nearest Neighbor

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:


print(accuracy_score(y_test,y_pred))


# In[ ]:


plot_roc_curve(knn,X_test,y_test)


# Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:


print(accuracy_score(y_test,y_pred))


# In[ ]:


plot_roc_curve(rf,X_test,y_test)


# Naive Bayes Classifier

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:


print(accuracy_score(y_test,y_pred))


# In[ ]:


plot_roc_curve(nb,X_test,y_test)


# # **SUMMARY**

# *Logistic Regression Model has Highest Accuracy : 88% Highest AUC Score : 0.94 and Least False Negative Rate*

# Therefore we will use Logistic Regression Model while classifying
