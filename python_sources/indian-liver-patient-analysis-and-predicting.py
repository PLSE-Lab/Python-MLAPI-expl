#!/usr/bin/env python
# coding: utf-8

# # Aim of the Exercise:
# 1. Cleaning data-Check for missing values in dataset
# 2. Data Visualization
# 3. To analyze data and study correlation and trends between different features of the dataset.
# 4. Discarding or incorporating new features in dataset
# 5. Finding best algorithm for predictive modelling

# In[67]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Reading the data

# In[2]:


data=pd.read_csv('../input/indian_liver_patient.csv')


# In[3]:


data.head()


# # Checking for the null or missing values in data

# In[9]:


data.isnull().sum()


# Since *Albumin_and_Globulin_Ratio* contains 4 null values,they either could be discarded or filled with suitable values. We are going to fill our values with mean of the column.

# In[12]:


data['Albumin_and_Globulin_Ratio'].fillna(data['Albumin_and_Globulin_Ratio'].mean(),inplace=True)


# # Statistical Parameters of the dataset

# In[18]:


data.describe(include='all')


# In[19]:


data.columns


# # Data Visualization
# We will try to draw folowing insights from the dataset:
#     1. How gender and age impacts the disease?
#     2. Is there any direct relation between 'Total_Bilirubin' ,'Direct_Bilirubin',
#        'Alkaline_Phosphotase' , 'Alamine_Aminotransferase',
#        'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
#        'Albumin_and_Globulin_Ratio'*
#     

# In[24]:


sns.countplot(label='count',x='Dataset',data=data);


# In[41]:


sns.catplot(data=data,y='Age',x='Gender',hue='Dataset',jitter=0.4);


# To me some features seem linearly correlated like Total_Bilirubin and Direct_Bilirubin, Aspartate_Aminotransferase and Alamine_Aminotransferase, Total_Protiens and Albumin.Going to perform jointplot for each of them.

# In[43]:


sns.jointplot("Total_Bilirubin", "Direct_Bilirubin", data=data, kind="reg")


# In[45]:


sns.jointplot("Aspartate_Aminotransferase", "Alamine_Aminotransferase", data=data, kind="reg")


# In[47]:


sns.jointplot("Total_Protiens", "Albumin", data=data, kind="reg")


# Since there seems a direct relationship between them ,so during features selection we can keep one feature from each of them.

# In[50]:


data.corr()


# # Results of Analysis:
# 1. Age and Gender affect the occurence of  disease.
# 2. Some features are directly correlated like Total_Bilirubin and Direct_Bilirubin, Aspartate_Aminotransferase and Alamine_Aminotransferase, Total_Protiens and Albumin.
# 3. Male has more the no of liver disease than female.

# # Feature Selection
# I am discarding Direct_Bilirubin,Aspartate_Aminotransferase.
# Features Kept:
# Total_Bilirubin
# Alamine_Aminotransferase
# Total_Protiens
# Albumin_and_Globulin_Ratio
# Albumin

# Since gender is categorical we need to convert it to numeric data.

# In[56]:


data = pd.concat([data,pd.get_dummies(data['Gender'], prefix = 'Gender')], axis=1)


# In[61]:


X = data.drop(['Gender','Dataset','Direct_Bilirubin','Aspartate_Aminotransferase'], axis=1)
y = data['Dataset']


# In[62]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


# # Model Predicting:
# I plan to train my model on different algorithms and check for best from them on basis of confusion matrix , accuracy etc.
# 1. Logistic Regression
# 2. SVM
# 3. Random Forests
# 4. Linear Regression

# # Logistic Regression

# In[64]:


logistic=LogisticRegression()
logistic.fit(X_train,y_train)
logispredicted=logistic.predict(X_test)
print('Training Score:',logistic.score(X_train, y_train))
print('Testing Score:',logistic.score(X_test, y_test))
print('Accuracy:',accuracy_score(y_test,logispredicted))
print('Confusion Matrix: \n', confusion_matrix(y_test,logispredicted))


# # SVM
# 

# In[68]:


svmclf = svm.SVC(gamma='scale')
svmclf.fit(X_train,y_train)
svmpredicted=logistic.predict(X_test)
print('Training Score:',svmclf.score(X_train, y_train))
print('Testing Score:',svmclf.score(X_test, y_test))
print('Accuracy:',accuracy_score(y_test,svmpredicted))
print('Confusion Matrix: \n', confusion_matrix(y_test,svmpredicted))


# # Random Forest

# In[65]:


# Random Forest

randomforest = RandomForestClassifier(n_estimators=100)
randomforest.fit(X_train, y_train)
#Predict Output
predicted = randomforest.predict(X_test)

print('Training Score:',randomforest.score(X_train, y_train))
print('Testing Score:',randomforest.score(X_test, y_test))
print('Accuracy:',accuracy_score(y_test,predicted))
print('Confusion Matrix: \n', confusion_matrix(y_test,predicted))


# # Linear regression

# In[70]:


linear = linear_model.LinearRegression()
linear.fit(X_train, y_train)
#Predict Output
lpredicted = linear.predict(X_test)

print('Training Score:',linear.score(X_train, y_train))
print('Testing Score:',linear.score(X_test, y_test))


# In[71]:


models = pd.DataFrame({
    'Model': [ 'Logistic Regression', 'SVM','Random Forest','Linear Regression'],
    'Score': [ logistic.score(X_train, y_train), svmclf.score(X_train, y_train), randomforest.score(X_train, y_train),linear.score(X_train, y_train)],
    'Test Score': [ logistic.score(X_test, y_test), svmclf.score(X_test, y_test), randomforest.score(X_test, y_test),linear.score(X_test, y_test)]})
models.sort_values(by='Test Score', ascending=False)


# In[ ]:




