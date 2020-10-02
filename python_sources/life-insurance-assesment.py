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


# Import Necessary Libraries & Import .csv files into pandas DataFrames 
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')


# In[ ]:


#Lets see data sample
data_train.sample(10)


# In[ ]:


# Lets check Df shape
data_train.shape

# there are 128 features.


# In[ ]:


data_test.shape


# **Feature details posted in data overview section - **
# The following variables are all categorical (nominal):
# 
# Product_Info_1, Product_Info_2, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, Employment_Info_2, Employment_Info_3, Employment_Info_5, InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, Insurance_History_1, Insurance_History_2, Insurance_History_3, Insurance_History_4, Insurance_History_7, Insurance_History_8, Insurance_History_9, Family_Hist_1, Medical_History_2, Medical_History_3, Medical_History_4, Medical_History_5, Medical_History_6, Medical_History_7, Medical_History_8, Medical_History_9, Medical_History_11, Medical_History_12, Medical_History_13, Medical_History_14, Medical_History_16, Medical_History_17, Medical_History_18, Medical_History_19, Medical_History_20, Medical_History_21, Medical_History_22, Medical_History_23, Medical_History_25, Medical_History_26, Medical_History_27, Medical_History_28, Medical_History_29, Medical_History_30, Medical_History_31, Medical_History_33, Medical_History_34, Medical_History_35, Medical_History_36, Medical_History_37, Medical_History_38, Medical_History_39, Medical_History_40, Medical_History_41
# 
# The following variables are continuous:
# 
# Product_Info_4, Ins_Age, Ht, Wt, BMI, Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5
# 
# The following variables are discrete:
# 
# Medical_History_1, Medical_History_10, Medical_History_15, Medical_History_24, Medical_History_32
# 
# Medical_Keyword_1-48 are dummy variables.

# We will check for missing values . 
# 
# If a categorical feature has missing values - if required will impute it with median 
# 
# if a continous feature has missing values - if required will impute it with mean 

# In[ ]:


data_train.dtypes
data_train.dtypes.unique()
#No string data type - all are numerical values which is good.


# 
# #Missing Value imputation 

# In[ ]:


data_train.isnull().sum()[data_train.isnull().sum() !=0]
#Below listed columns have missing values in the combined (Train+test) dataset. 


# In[ ]:


# Lets draw a bar graph to visualize percentage of missing features in train set
missing= data_train.isnull().sum()[data_train.isnull().sum() !=0]
missing=pd.DataFrame(missing.reset_index())
missing.rename(columns={'index':'features',0:'missing_count'},inplace=True)
missing['missing_count_percentage']=((missing['missing_count'])/59381)*100
plt.figure(figsize=(20,8))
sns.barplot(y=missing['features'],x=missing['missing_count_percentage'])

#Looking at below bar grah- 
#Medical_Hist_32/24/15/10 , Family_hist_5 are top five features with huge amount of missing data ( imputaion to these might not be fruitful - I will drop these features)


#  
#  
#  Employment_Info_1_4_6 Insurance_History_5 Family_Hist_2-3-4-5 are continous features . 
# 
# The following variables are discrete:
# Medical_History_1, Medical_History_10, Medical_History_15, Medical_History_24, Medical_History_32
# 
# 1. remove rows with missing values and see model performance 
# 2. impute missing values with mean and median or may be mode.
# 
# 

# In[ ]:


# Lets see spread of data before we impute missing values
plt.plot(figsize=(15,10))
sns.boxplot(data_train['Employment_Info_1'])
# Employment_Info_1 seems to have lots of outliers - Median should be right to impute missing values


# In[ ]:


data_train['Employment_Info_1'].isna().sum()


# In[ ]:


data_train['Employment_Info_1'].fillna(data_train['Employment_Info_1'].median(),inplace=True) 
# imputing with Meadian , as there are lots of Outliers 
data_test['Employment_Info_1'].fillna(data_test['Employment_Info_1'].median(),inplace=True) 


# In[ ]:


data_train['Employment_Info_1'].isna().sum()


# In[ ]:


#Outlier Treatment -
data_train['Employment_Info_1'].describe()


# In[ ]:


sns.boxplot(data_train['Employment_Info_4'])
# ['Employment_Info_4'] is has most of the values centered close to zero , also huge presence of outliers 


# In[ ]:


data_train['Employment_Info_4'].fillna(data_train['Employment_Info_4'].median(),inplace=True)
data_test['Employment_Info_4'].fillna(data_test['Employment_Info_4'].median(),inplace=True)


# In[ ]:


sns.boxplot(data_train['Employment_Info_6'])
#No outlieers - mean should be rigth candidate to impute missing values


# In[ ]:


data_train['Employment_Info_6'].fillna(data_train['Employment_Info_6'].mean(),inplace=True)
data_test['Employment_Info_6'].fillna(data_test['Employment_Info_6'].mean(),inplace=True)


# In[ ]:


sns.boxplot(y=data_train['Medical_History_1'])


# In[ ]:


data_train['Medical_History_1'].fillna(data_train['Medical_History_1'].median(),inplace=True)
data_test['Medical_History_1'].fillna(data_test['Medical_History_1'].median(),inplace=True)


# In[ ]:


#lets drop features with high number of missing values 
data_train.drop(['Medical_History_10','Medical_History_15','Medical_History_24','Medical_History_32','Family_Hist_3','Family_Hist_5','Family_Hist_2','Family_Hist_4'],axis=1,inplace=True)


# In[ ]:


data_test.drop(['Medical_History_10','Medical_History_15','Medical_History_24','Medical_History_32','Family_Hist_3','Family_Hist_5','Family_Hist_2','Family_Hist_4'],axis=1,inplace=True)


# In[ ]:


data_train.isnull().sum()[data_train.isnull().sum()!=0]


# In[ ]:


#imputing with median 
data_train['Insurance_History_5'].fillna(data_train['Insurance_History_5'].median(),inplace=True)
data_test['Insurance_History_5'].fillna(data_test['Insurance_History_5'].median(),inplace=True)


# In[ ]:


data_train.isnull().sum()
#All missing NA values has been treated


# #Now that we have imputed Missing values - we can move to next step to convert string type feature data into numric data

# In[ ]:


data_train.head()
#Product_info_2 seems to be the only feature where we should map string values with numeric categorical values


# In[ ]:


data_train['Product_Info_2'].unique()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data_train['Product_Info_2']=le.fit_transform(data_train['Product_Info_2'])
data_test['Product_Info_2']=le.transform(data_test['Product_Info_2'])

#data_train.dtypes
#Employment_Info_1-4-6  Insurance_History_5
# I faced an error stating dta types of train columns are not float/numeric ill apply encoder on all column and see what happens


# In[ ]:


data_train.head()


# In[ ]:


# feature meatrix and response vector seperation
X_train=data_train.iloc[:,0:-1]
y_train=data_train['Response']
X_train.drop('Id',axis=1,inplace=True)


# In[ ]:


X_train.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_train,y_train)


# Machine Learning Model fitting and prediction

# In[ ]:


y_train.unique()
#there are 8 labels/class in dataset


# In[ ]:



from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.multiclass import OneVsRestClassifier


# In[ ]:


# Using a Decision Tree classifier 
from sklearn.tree import DecisionTreeClassifier
param_grid={'max_depth':range(1,20,2)}
DT=DecisionTreeClassifier()
clf_DT=GridSearchCV(DT,param_grid,cv=10,scoring='accuracy',n_jobs=-1).fit(X_train,y_train)
y_pred=clf_DT.predict(X_test)
print(accuracy_score(y_test,y_pred))


# In[ ]:


#Using a Random Forest tree classifier
from sklearn.ensemble import RandomForestClassifier
param_grid={'max_depth':range(1,20,2)}
RF=RandomForestClassifier()
clf_rf=GridSearchCV(RF,param_grid,cv=10,scoring='accuracy',n_jobs=-1).fit(X_train,y_train)
y_pred=clf_rf.predict(X_test)
accuracy_score(y_test,y_pred)


# For now i'll use Decison tree for summission ,  i'll work on to improve my predictions, any suggestion/feedback is appreciated.
# 

# In[ ]:


ids = data_test['Id']
predictions = clf_DT.predict(data_test.drop('Id', axis=1))


output = pd.DataFrame({ 'Id' : ids, 'Response': predictions })
output.to_csv('/Users/adityaprakash/Downloads/predictions.csv', index = False)
output.head()


# In[ ]:




