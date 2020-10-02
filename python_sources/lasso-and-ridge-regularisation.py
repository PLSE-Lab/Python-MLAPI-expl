#!/usr/bin/env python
# coding: utf-8

# # Lasso Regularisation

# **Regularization** consists in adding a penalty on the different parameters of the model to reduce the freedom of the model. Hence, the model will be less likely to fit the noise of the training data and will improve the generalization abilities of the model.  **Lasso** or** L1 **has the property that is able to shrink some of the coefficients to zero. It adds penalty  equivalent to absolute value of the magnitude of coefficients. Therefore, that feature can be removed from the model.

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso,LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input"))


# I will be using **santander customer satisfaction** dataset since I have used the same dataset in [Filter Feature Selection method](https://www.kaggle.com/raviprakash438/filter-method-feature-selection/notebook). It will be easy to compare the scores of both feature selection methods.

# In[2]:


#Load the train dataset. It contain more then 76000 records. Lets load 10000 records only to make things fast.
df=pd.read_csv('../input/santander-customer-satisfaction/train.csv',nrows=10000)
df.shape


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


# separate dataset into train and test
X_train, X_test, Y_train, Y_test = train_test_split(df.drop(labels=['TARGET'], axis=1),df['TARGET'],test_size=0.3,random_state=0)
#Filling null value with 0.
X_train.fillna(0,inplace=True)
X_test.fillna(0,inplace=True)
#Shape of training set and test set.
X_train.shape, X_test.shape


# In[6]:


# linear models benefit from feature scaling
scaler=StandardScaler()
scaler.fit(X_train)


# In[13]:


#Lets do the model fitting and feature selection all in single line of code.
#I will be using Logistic Regression model and select Lasso (l1 as) as a penalty
#I will be using SelectFromModel object which select the features which are non zero.
#C=1 (Inverse of regularization strength.Smaller values specify stronger regularization.)
#penalty='l1' (Specify the norm used in the penalization.Here we are using Lasso.)
sel=SelectFromModel(LogisticRegression(C=1,penalty='l1'))
sel.fit(scaler.transform(X_train),Y_train)


# In[14]:


print('Total features-->',X_train.shape[1])
print('Selected featurs-->',sum(sel.get_support()))
print('Removed featurs-->',np.sum(sel.estimator_.coef_==0))


# As we can see, we have used Lasso regularisation to remove non important features from the dataset. If we compare Lasso regularisation with [Filter Method](https://www.kaggle.com/raviprakash438/filter-method-feature-selection/notebook), there we used Constant, Quasi-Constant, Duplicate and Correlation methods to remove the non important features but here we have done all those things in just two lines. Isn't that great? But we should keep in mind that increasing the penalisation will remove more features.

# In[9]:


# create a function to build random forests and compare performance in train and test set
def RandomForest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=200, random_state=1, max_depth=4)
    rf.fit(X_train, y_train)
    print('Train set')
    pred = rf.predict_proba(X_train)
    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
    print('Test set')
    pred = rf.predict_proba(X_test)
    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# In[10]:


#Transforming the training set and test set.
X_train_lasso=sel.transform(X_train)
X_test_lasso=sel.transform(X_test)
RandomForest(X_train_lasso,X_test_lasso,Y_train,Y_test)


# Now we will compare the result of Filter method and Lasso Regularisation. Please click [Filter Method](https://www.kaggle.com/raviprakash438/filter-method-feature-selection/notebook) to see the result. <br>
# Filter Method Score = 0.76199<br>
# Lasso Regularisation Score = 0.76250 <br>
# So, the result is almost similar. Therefore, for this dataset any of two approaches will give the similar result.

# # Ridge Regularisation

# **Ridge** or **L2** regression is the most commonly used method of regularization for the  problems which  do not have a unique solution. It adds penalty equivalent to square of the magnitude of coefficients.  Unlike L1 it don't srink some of the coefficients to zero. It srink the coefficients near to zero but it never by zero.

# In[23]:


#Lets do the model fitting and feature selection all in single line of code.
#I will be using Logistic Regression model and select Lasso (l1 as) as a penalty
#I will be using SelectFromModel object which select the features which are non zero.
#C=1 (Inverse of regularization strength.Smaller values specify stronger regularization.)
#penalty='l2' (Specify the norm used in the penalization.Here we are using Ridge. It is a default penalty.)
sfm=SelectFromModel(LogisticRegression(C=1,penalty='l2'))
sfm.fit(scaler.transform(X_train),Y_train)


# In[24]:


print('Total features-->',X_train.shape[1])
print('Selected featurs-->',sum(sfm.get_support()))
print('Removed featurs-->',np.sum(sfm.estimator_.coef_==0))


# As I told that L2 or Ridge regression will not srink the coefficient to zero but here we are able to see 86 features have zero coefficient. L2 have not srink the coefficient of the features to zero. Actually, these are [constant features](https://www.kaggle.com/raviprakash438/filter-method-feature-selection) which means these features have same value for all samples. You can checkout my [Filter method feature selection post](https://www.kaggle.com/raviprakash438/filter-method-feature-selection) there also I have used same dataset for feature selection . <br>
# 
# You will thinking on what basis the selected feature count is 107 out of 370. Actually it is selecting those coefficients whose absolute value is greater than absolute coefficient mean  as shown below.

# In[25]:


np.sum(np.abs(sfm.estimator_.coef_)>np.abs(sfm.estimator_.coef_).mean())


# In[26]:


#Transforming the training set and test set.
X_train_l2=sel.transform(X_train)
X_test_l2=sel.transform(X_test)
RandomForest(X_train_l2,X_test_l2,Y_train,Y_test)


# Now we will compare the result of Filter method, Lasso Regularisation and Ridge Regularisation. Please click [Filter Method](https://www.kaggle.com/raviprakash438/filter-method-feature-selection/notebook) to see the result. <br>
# Filter Method Score = 0.76199<br>
# Lasso Regularisation Score = 0.76250 <br>
# Ridge Regularisation Score = 0.76090 <br>
# So, the result is almost similar. Therefore, for this dataset any of three approaches will give the similar result.

# **Please checkout [Feature Selection Main Page](https://www.kaggle.com/raviprakash438/feature-selection-technique-in-machine-learning)**

# ***Please share your comments,likes or dislikes so that I can improve the post.***

# In[ ]:




