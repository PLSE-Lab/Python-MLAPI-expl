#!/usr/bin/env python
# coding: utf-8

# # Notebook Description:
#    In this Notebook we will analyze the Ad success with and without oversampling(SMOTE). Then we will see the influence of oversampling.
#   1. Imabalanced dataset and SMOTE
#   2. Preprocessing
#   3. Logistic Regression without SMOTE
#   4. Logistic Regression with SMOTE
#     
# # How to measure an imbalanced dataset?
# 
#     When the dependent variable is categorical and the classes are not in equal proportion in the data, then such datasets can be termed as imbalanced dataset. Mostly in case of anomaly detection   the dependent variables will be imbalanced.
#     
#     How to select the metrics? Accuracy is not a right metrics, because if we mark every results to the majority  class (No Succes in case of ad's sucess) then we will have accuracy, but the results are not as expected. So in such cases it is better to know how well the minority class (Success) was evaluated. So we can determine the precision.
#     
#    precision = TP / (TP + FP) 
#     
# # SMOTE
# 
#     We have a metrics to measure, still what if the minority class is  very less (very less success). In that case we want the model to learn more about the minority class. (Like you concentrate on the topic you are not strong in your school exam). This process is called over sampling. Sythetic Minority Oversampling Technique (SMOTE). 
#     
#    How SMOTE works? 
#              
#     We need to be aware how much samples we need for oversampling.
#        * Choose the Minority Class
#        * Calculate the no of samples from minority class required.
#        * For each sample to be synthesized,
#           - Randomly choose a minor class instance in the feature space.(say pt 'a')
#           - Select k_nearest neighbours to the instance.
#           - Choose one of the k nearest neighbours at random ('b').
#           - Join 'a' & 'b' in the feature space by a line.
#           - select a random point on the line. This will be the sample sythesized.
#        
#    For more refer: https://www.youtube.com/watch?v=U3X98xZ4_no
#           
#    How do we implement SMOTE?
#        
#        SMOTE can be applied to a dataset in python using imbalanced_learn library.
#    

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

get_ipython().system('pip install category_encoders')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualization library
import seaborn as sns # visualization library

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/advertsuccess/Train.csv", index_col='id')


# In[ ]:


data.info()


# In[ ]:


data.netgain.value_counts().plot(kind='barh')
plt.title('Net gain ( \'True\' or \'False\' )', fontsize=20)
plt.show()


#     We see that total data we have 26,048 entries, of which 6195 have a net gain. This is 23% and so the dataset is imbalanced. Here we can construct a model and evaluate using prceision. We will  do preprocessing and construct a simple logistic regression.
#     
#     Then we can apply SMOTE to our data and analyze our data in the similar fashion.

# # Preprocessing
#      
#    There are 11 columns of which 9 are categorical. You expand the cells to find visualization of distribution of columns. There are 3 binary columns which will be binary encoded and other categoricals will be One Hot Encoded. Then the dataset will be standardized. In second case we will apply SMOTE before standardizing.
#    
#    1) Encoding -> Standardization -> Training -> Precision
#    
#    2) Encoding -> **SMOTE** -> Standardization -> Training -> Precision

# In[ ]:


data.head()


# In[ ]:


plt.figure(figsize=(15,4))
plt.title('Relationship Status',fontsize=20)
sns.countplot(data.realtionship_status)
plt.show()


# In[ ]:


plt.figure(figsize=(15,4))
plt.title('Industry',fontsize=20)
sns.countplot(data.industry)
plt.show()


# In[ ]:


plt.figure(figsize=(15,4))
plt.title('Genre',fontsize=20)
sns.countplot(data.genre)
plt.show()


# In[ ]:


plt.figure(figsize=(15,4))
plt.hist(data['average_runtime(minutes_per_week)'],bins=25)
plt.title('average_runtime(minutes_per_week)',fontsize=20)
plt.show()


# In[ ]:


plt.figure(figsize=(15,4))
plt.title('Gender',fontsize=20)
sns.countplot(data.targeted_sex)
plt.show()


# In[ ]:


plt.figure(figsize=(15,4))
plt.title('Air time',fontsize=20)
sns.countplot(data.airtime)
plt.show()


# In[ ]:


plt.figure(figsize=(5,10))
plt.title('Air Location',fontsize=20)
data.airlocation.value_counts().plot(kind='barh')
plt.show()


# In[ ]:


plt.figure(figsize=(15,4))
plt.title('Expensive',fontsize=20)
sns.countplot(data.expensive)
plt.show()


# In[ ]:


plt.figure(figsize=(15,4))
plt.title('Money back Guarantee - Yes or No',fontsize=20)
sns.countplot(data.money_back_guarantee)
plt.show()


# In[ ]:


#Preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from category_encoders import BinaryEncoder
from sklearn.metrics import precision_score
#expensive is an ordinal categorical variable
exp_dict = {'Low':0,'Medium':1,'High':2}
data['expensive'] = data.expensive.map(exp_dict)

#Binary Categorical
Bin_columns = ['targeted_sex','money_back_guarantee']
Bin_Encoder = BinaryEncoder()

#Multi class nominal categorical
cat_columns = ['realtionship_status', 'industry', 'genre', 'airtime', 'airlocation' ]
OHE = OneHotEncoder(sparse=False)

encoding = ColumnTransformer(transformers=[('cat',OHE,cat_columns),
                                               ('bin',Bin_Encoder,Bin_columns)])

clf = Pipeline(steps=[('encoder',encoding),('Std',StandardScaler()),('LR',LogisticRegression())])

y, X = data['netgain'],data.drop('netgain',axis=1)

X_train, X_test, y_train,  y_test = train_test_split(X,y)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

precision_score(y_pred,y_test)


# In[ ]:


# Applying SMOTE
from imblearn.over_sampling import SMOTE 

#SMOTE have to be applied for training set only

preprocessor = Pipeline(steps=[('encoder',encoding),('Std',StandardScaler())])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

print("Shape of train dataset before applying SMOTE:",X_train.shape)

X_train, y_train = SMOTE().fit_resample(X_train,y_train)

print("Shape of train dataset after applying SMOTE:",X_train.shape)


# Now we have 10000 more samples! These are synthesized from the minority class.
# 
# Lets make logistic regression for the resampled data.

# In[ ]:


lr2 = LogisticRegression()
lr2.fit(X_train,y_train)
y_pred2 = lr2.predict(X_test)
precision_score(y_pred2,y_test)


# *Here the Precision of our classification has improved by nearly 25%*
# 
# 
# * Feedback, upvotes are most welcomed! *

# REFERENCE:
# 
#   https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
#   https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
