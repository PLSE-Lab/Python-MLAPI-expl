#!/usr/bin/env python
# coding: utf-8

# In this kernel we will see how to do Feature Selection.We will be using Lasso,Ridge and Embeded technique.Regularization means adding penalty to the parameters of the machine learning model.THis helps in reducing the freedom of the model.Lasso regression has the ability to shrink the coefficinets to zero.This helps in carrying out feature selection.This Kernel is a work in process and I will be updating the kernel in coming days.If you like my work please do vote.

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


# **Importing the Python Modules**

# In[ ]:


df = pd.read_csv('../input/paribas-claim-feature-selection/train.csv',nrows=50000)
df.head()


# In[ ]:


import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Lasso,LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler


# We have imported only 50000 rows of data and we can see that there are 133 feature in our dataset.Out task in to used feature selection and to reduce the number of feature needed by our machine learning model to predic the outcome.

# In[ ]:


# Creating a copy of the data 
df_copy = df.copy()
df.shape


# In[ ]:


# In practise feature selection should be done after pre processing the data categorical data should be encoded and then only we need to access how deterministic they are of the target
# Here we will be considering Numerical Variables 
# Selecting the numerical columns with the below lines of code 

numerics = ['int16','int32','int64','float16','float32','float64']
numerical_vars = list(df.select_dtypes(include = numerics).columns)
df = df[numerical_vars]
df.shape


# So intially we had 133 features after removing the categorically data we have only 114 numerical data.

# Pandas Profiling to get Stats

# ### 1.Lasso Regularization

# **Test Train Split**
# 
# It is a good practise to do feature selection only on the training dataset.

# In[ ]:


# seperate train and test sets 
X_train,X_test,y_train,y_test = train_test_split(df.drop(labels=['target','ID'],axis=1),df['target'],test_size=0.3,random_state=0)
X_train.shape,X_test.shape


# **Feature Scaling**

# In[ ]:


# Linear Model Benifits with feature Scaling

scaler = StandardScaler()
scaler.fit(X_train.fillna(0))


# In[ ]:


# We will be doing model fitting and feature selection together 
# We will specify Logistic regression and select Lasso (L1) penalty 
# SelectFromModel from sklearn will be used to slect the features for which coefficients are non-zero

#sel_ = SelectFromModel(LogisticRegression(C=1, penalty= 'l1'))

sel_ = SelectFromModel(Lasso(alpha=0.005,random_state=0))
sel_.fit(scaler.transform(X_train.fillna(0)), y_train)

# I used penalty as none as l1 option was not working


# In[ ]:


# this command lets us to vizualise which features were kept
sel_.get_support()


# In[ ]:


# We can now make a list of selected features 
selected_feat = X_train.columns[(sel_.get_support())]

print('Total features:{}'.format((X_train.shape[1])))
print('Selected features:{}'.format(len(selected_feat)))
print('Features with coefficients shrank to zero:{}'.format(np.sum(sel_.estimator_.coef_==0)))


# By Changing the Value of alpha we can change the number of selected features.Hence we can try some iteration to arrive at optimum value of number of selected features

# In[ ]:


# The number of features which coefficient was shrank to zero 
np.sum(sel_.estimator_.coef_==0)


# In[ ]:


# Identifying the removed features 
removed_feats =X_train.columns[(sel_.estimator_.coef_==0).ravel().tolist()]
removed_feats


# In[ ]:


selected_feat


# In[ ]:


df.shape


# In[ ]:


# We can now remove the features from training and test set 
X_train_selected = sel_.transform(X_train.fillna(0))
X_test_selected = sel_.transform(X_test.fillna(0))
X_train_selected.shape,X_test_selected.shape


# **Logistic Regression**

# In[ ]:


"""from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train_selected,y_train)"""
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train_selected,y_train)


# **Making the Prediction **

# In[ ]:


#y_test=lm.predict(X_test_selected)
y_pred=classifier.predict(X_test_selected)
print(y_test)


# ### Model Performance

# **A].Accuracy**

# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[ ]:


print('Accuracy of model is:',accuracy_score(y_test,y_pred))


# **B].Confusion Matrix**

# In[ ]:


from sklearn.metrics import confusion_matrix  #Class has capital at the begining function starts with small letters 
cm=confusion_matrix(y_test,y_pred)
import seaborn as sns
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.title("Test for Test Dataset")
plt.xlabel("predicted y values")
plt.ylabel("real y values")
plt.show()


# **C].Classification Report **

# In[ ]:


print(classification_report(y_test,y_pred))


# ### 2.Ridge Regularization 
# 
# L2 regularization doesnt sink the coefficients to Zero

# In[ ]:


df1=df_copy
df1.shape


# **Getting only Numerical Variables**

# In[ ]:


# In practise feature selection should be done after pre processing the data categorical data should be encoded and then only we need to access how deterministic they are of the target
# Here we will be considering Numerical Variables 
# Selecting the numerical columns with the below lines of code 

numerics = ['int16','int32','int64','float16','float32','float64']
numerical_vars = list(df1.select_dtypes(include = numerics).columns)
df1 = df1[numerical_vars]
df1.shape


# In[ ]:


df.shape


# From 133 the number of variables has gone done to 114 as we are considering only Numerical Variables.

# **Test Train Split**

# In[ ]:


# seperate train and test sets 
X_train,X_test,y_train,y_test = train_test_split(df1.drop(labels=['target','ID'],axis=1),df1['target'],test_size=0.3,random_state=0)
X_train.shape,X_test.shape


# In[ ]:


# Linear Model Benifits with feature Scaling

scaler = StandardScaler()
scaler.fit(X_train.fillna(0))


# **Logistic Regression with Selection Algorithm**

# In[ ]:


sel_ = SelectFromModel(LogisticRegression(C=1000,penalty='l2'))
sel_.fit(scaler.transform(X_train.fillna(0)), y_train)


# In[ ]:


# We will be doing model fitting and feature selection together 
# We will specify Logistic regression and select Lasso (L1) penalty 
# SelectFromModel from sklearn will be used to slect the features for which coefficients are non-zero

#sel_ = SelectFromModel(LogisticRegression(C=1, penalty= 'l1'))

sel_ = SelectFromModel(Lasso(alpha=0.005,random_state=0))
sel_.fit(scaler.transform(X_train.fillna(0)), y_train)

# I used penalty as none as l1 option was not working


# In[ ]:


sel_.get_support()


# The features with False have been removed by the feature selection algorithm.Here the coefficents are selected based on the mean value of the coefficients.The features with coefficent values higer than the mean values are retained.

# In[ ]:


selected_feat = X_train.columns[(sel_.get_support())]
len(selected_feat)


# So out of 112 we have finally ended up with 60 features based on Mean value of the coefficients

# In[ ]:


np.sum(sel_.estimator_.coef_==0)


# In[ ]:


sel_.estimator_.coef_.mean()


# So we can see that Ridge Reglarization doesnt make the coefficient of the features to Zero.We have done features based on the mean value of the coefficients.The features having mean coefficent value more than 0.00889 will be selected.

# **Distribution of Coefficients**

# In[ ]:


pd.Series(sel_.estimator_.coef_.ravel()).hist();


# So the distribution shows that we have positive and negative values of the coefficints.They show that some fatures have positive and some have negative correlation with the Target.
# 
# The absolute value of the coefficients give an idea about the importance of the feature.So Feature selection will be don by filtering on the absolute value of the oefficients. 

# In[ ]:


np.abs(sel_.estimator_.coef_).mean()


# **Plotting the historgram with the absoulte value of the coefficient **

# In[ ]:


pd.Series(np.abs(sel_.estimator_.coef_).ravel()).hist();


# In[ ]:


# Comparing the number of selected features with the coefficients who have value above the mean of thw absoulte value of the coefficents
print('Total features: {}'.format((X_train.shape[1])))
print('Selected features: {}'.format(len(selected_feat)))
print('Features with coefficients greater than the mean coefficient:{}'.format(np.sum(np.abs(sel_.estimator_.coef_)>np.abs(sel_.estimator_.coef_).mean())))


# So Selected features is matching the number of coefficients with absolute value greater than the mean of absolute value of the coeffients.So our method of feature selection is correct.

# In[ ]:




