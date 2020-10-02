#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
# importing plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
# Splits the dataframe into random train and test subsets to to 
# train the model and test
from sklearn.model_selection import train_test_split
# Gaussian Naive bayes for model building
from sklearn.naive_bayes import GaussianNB
# Metrics for accuracy and confusion matrix and classification report
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler,scale


# In[2]:


# To enable plotting graphs
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Read the data from CSV using pandas 
df = pd.read_csv('../input/pima-indians-diabetes_nb.csv')


# In[4]:


# Check data is read into pandas dataframe
df.head()


# In[ ]:


# Check for number rows and columns present in the dataset
df.shape


# In[ ]:


# To check data type of columns in the dataframe
df.info()


# In[ ]:


# Check for null values
df.isnull().any()


# In[ ]:


df[~df.applymap(np.isreal).all(1)]


# In[ ]:


# Check target column 'class' count 
df['class'].value_counts()


# In[ ]:


# See the plot for categorical target column count
sns.countplot(df['class'])


# In[ ]:


# Distribution of the attributes in the dataset
df.describe().T


# In[ ]:


# See the histograms for each independant variable present in dataframe df


# In[ ]:


# min is zero and right skewed
df['Preg'].hist()


# In[ ]:


# Left skewed
df['Plas'].hist()


# In[ ]:


# left skewed and outliers present
df['Pres'].hist()


# In[ ]:


# min zero more right skewed
df['skin'].hist()


# In[ ]:


# Large right skewed
df['test'].hist()


# In[ ]:


# Mean and median almost near 
df['mass'].hist()


# In[ ]:


# mean and median little near
df['pedi'].hist()


# In[ ]:


# data points largely right skewed 
sns.distplot(df['age'])


# In[ ]:


df['class'].hist()


# In[ ]:


sns.pairplot(df, hue = 'class')


# In[ ]:


# When observe the pairplot plots and distribution of data (Plas,Pres, skin and mass) are nearly normally distributed
# Attributes (Preg, test, pedi ) has exponential distribution


# In[ ]:


sns.boxplot(x = df['Preg'] )


# In[ ]:


sns.boxplot(x = df['test'] )


# In[ ]:


df.isnull().any()


# In[ ]:


corr = df.corr()


# In[ ]:


sns.heatmap(corr,annot = True)


# In[ ]:


df.groupby('class').hist(figsize=(9, 9))


# In[ ]:


df1 = df


# In[ ]:


# There are few zero values as this attribute is contnuous consider as missing values
#df1['Plas'] = df1['Plas'].replace(0,np.NaN)


# In[ ]:


#df1['Plas'] = df1['Plas'].replace(np.NaN, df1['Plas'].mean(skipna=True))


# In[ ]:


df1.head()


# In[ ]:


# drop target value for X values independent attributes
X = df1.drop(['class'], axis =1)
# target attribute - dependant attribute
y = df1['class']


# In[ ]:


# split the data 30 %
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 1)


# In[ ]:


# Gaussian Naivebayes model
gnb = GaussianNB()


# In[ ]:


# train the model with train split
gnb.fit(X_train,y_train)


# In[ ]:


# Check the score for train data
gnb.score(X_train,y_train)


# In[ ]:


# Predict the target value for test data
y_predict = gnb.predict(X_test)


# In[ ]:


# Check the score for test data 
gnb.score(X_test,y_test)


# In[ ]:


# see the confusion matrics for precision and recall
print(metrics.confusion_matrix(y_test,y_predict))


# In[ ]:


# see the f1 score and precision , recall percentage
print(metrics.classification_report(y_test,y_predict))


# In[ ]:


print(metrics.accuracy_score(y_test,y_predict))


# Analyzing the confusion matrix
# 
# True Positives (TP): we correctly predicted that they do have diabetes 53
# 
# True Negatives (TN): we correctly predicted that they don't have diabetes 128
# 
# False Positives (FP): we incorrectly predicted that they do have diabetes (a "Type I error") 18 Falsely predict positive Type I error
# 
# False Negatives (FN): we incorrectly predicted that they don't have diabetes (a "Type II error") 32 Falsely predict negative Type II error
# 

# # Improve the model -----------------------------Iteration 2 -----------------------------------------------

# In[ ]:


X = df1.drop(['class'], axis =1)


# In[ ]:


y = df1['class']


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 1)


# In[ ]:


from sklearn.preprocessing import scale
X_train_scale = scale(X_train)
X_test_scale = scale(X_test)


# In[ ]:


X_test_scale.shape


# In[ ]:


X_train_scale.shape


# In[ ]:


X_train_scale


# In[ ]:


gnb = GaussianNB()


# In[ ]:


gnb.fit(X_train_scale,y_train)


# In[ ]:


gnb.score(X_train_scale,y_train)


# In[ ]:


y_predict = gnb.predict(X_test_scale)


# In[ ]:


gnb.score(X_test_scale,y_test)


# In[ ]:


print(metrics.classification_report(y_test,y_predict))


# In[ ]:


print(metrics.confusion_matrix(y_test,y_predict))


# In[ ]:


print(metrics.accuracy_score(y_test,y_predict))


# # Improve the model -----------------------------Iteration 2 -----------------------------------------------

# In[ ]:


X = df1.drop(['class'], axis =1)


# In[ ]:


y = df1['class']


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 1)


# In[ ]:


scale = StandardScaler()


# In[ ]:


X_train_scale = scale.fit_transform(X_train)
X_test_scale = scale.fit_transform(X_test)


# In[ ]:


X_test_scale.shape


# In[ ]:


X_train_scale.shape


# In[ ]:


X_train_scale


# In[ ]:


gnb = GaussianNB()


# In[ ]:


gnb.fit(X_train_scale,y_train)


# In[ ]:


gnb.score(X_train_scale,y_train)


# In[ ]:


y_predict = gnb.predict(X_test_scale)


# In[ ]:


gnb.score(X_test_scale,y_test)


# In[ ]:


print(metrics.classification_report(y_test,y_predict))


# In[ ]:


print(metrics.confusion_matrix(y_test,y_predict))


# In[ ]:


print(metrics.accuracy_score(y_test,y_predict))

