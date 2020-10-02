#!/usr/bin/env python
# coding: utf-8

# ## Predicting Bank Telemarketing outcomes ##

# In[2]:


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


# In[72]:


#make appropriate imports 
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
import math
import seaborn as sns

import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# ### Reading in Data ##
# 
# We are using data collected from a Portuguese retail bank, from May 2008 to June 2013, in a total of 52,944 phone contacts.This data is collected from the agents executing phone calls to a list of clients to sell the deposit. Thus,the result is a binary unsuccessful or successful contact of selling deposits.We have many variables which are taken into account (parameters)

# In[74]:


#read in train and test data and split 
train_data = pd.read_csv("../input/bank-train.csv")
test_data = pd.read_csv("../input/bank-test.csv")
train_data
X = train_data.iloc[:,:21]
y = train_data.iloc[:,21:]
#only x values, no predicted values so no need to split 
X_test = test_data


# ## Data Exploration##

# In[76]:


sns.heatmap(X.corr(), cmap="coolwarm")
print("Checking for correlations among variables")


# We use a heatmap to check for correlation among variables, and remove variables that are highly correlated when performing logistic regression.
# <br> We would theoretically want to drop euribo3m & emp.var.rate due to high correlation (violates assumption for log), but we dont.

# ## Class imbalance

# In[81]:


train_data["Client Subscription"] = train_data["y"].replace({1: "subscribed", 0: "did not subscribe"})
sns.countplot(train_data["Client Subscription"], palette = "BrBG")
plt.title("Did the client subscribe?")
print("percentage of clients subscribed: {}%".format(np.round(train_data.y.mean()*100, 2)))


# In[85]:


sns.countplot(train_data["month"], hue = train_data["Client Subscription"], palette = "BrBG")


# ## Preprocessing data ##

# In[7]:


from sklearn import preprocessing
#get just the categorical data of train_data 
cat_data=  X.select_dtypes(include=[object])
cat_data.head(3)


# Having so many categorical variables, we dont want to just ignore them. ML algorithms cant typically take in these values, so we convert to numerical values using pandas dummy varaible method or one hot encoding from sklearn

# In[8]:


#Converting all categorical var to numbers by making each option a column
#can increase dimensionality a lotttt if lots of variety in categorical options 
#month added 12 more columns for example for each month 
#can also do this with one hot encoding 
print('Size Before changing to dummy: ' + str(cat_data.shape))
cat_data = pd.get_dummies(cat_data)
print('Size After changing to dummy: ' + str(cat_data.shape))
cat_data.head(3)


# In[9]:


#numerical aspects of the data for training dset
#merge with categorical to make final X train data set
num_data= X.select_dtypes(include=[int, float])
merged = pd.concat([cat_data,num_data], axis=1)
X = merged
print(X.shape)


# We do the same preprocessing to the Test data 

# In[10]:


#change test data the same way 
num2 = X_test.select_dtypes(include=[int, float])
cat_for_test=  X_test.select_dtypes(include=[object])
X_2_numerical = pd.get_dummies(cat_for_test)
merged2 = pd.concat([X_2_numerical,num2], axis=1)


# ## Logistic Regression with all features ##
# 
# We hypothesize that this will result in a low F score as the model is probably overft
# 

# In[69]:


from sklearn.linear_model import LogisticRegression
#first we do a logistic regression using all variables 
all_features = list(merged2.columns.values)
x_train = X[all_features]
y_train = y
# create and fit model
LogReg = LogisticRegression(solver='newton-cg')
LogReg.fit(x_train, y_train)


# In[71]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#make predictions about what our dependent values would be for training set(yes or no)
y_train_pred = LogReg.predict(x_train)
#make a confusion matrix to see results of how our model performs 
confusion_matrix = confusion_matrix(y_train, y_train_pred)
print(classification_report(y_train, y_train_pred))


# In[23]:


#finally make predictiosn about our unknown test data
x_test = merged2[all_features]
predictions = LogReg.predict(x_test)
predictions


# In[105]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("Precision: " + str(precision_score(y_train_pred, y_train, average='micro')))
print("Recall: " + str(recall_score(y_train_pred, y_train, average='micro')))
print("F-1 Training Score: " + str(f1_score(y_train_pred, y_train, average='micro')))

print("F-1 Test Score based on results of submission: " + str(0.90651) )


# ## Using Feature Selection##
# 
# We Expect to get a better metric as we are only using the best features to avoid bias 

# Now we want to try to do some feature selection and see if there are certain features which may be more useful for making our model as sometimes adding too many features can cause overfitting. 
# 

# In[15]:


from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest,f_classif
#we take the top 10 best features (using k = 11 because we disregard the first feature duration)
#we are doing f-test statistic to find each score to determine most relevent feature  
bestfeatures = SelectKBest(score_func=f_classif, k=11)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(11,'Score'))  #print 11 best features (take 10)

#our top 10 best features we will model with 
best_features = ["nr.employed","pdays","poutcome_success","euribor3m","emp.var.rate","id","previous","poutcome_nonexistent","month_mar","contact_cellular"]


# In[65]:


#Ten Best 
x_train_best10 = X[best_features]
y_train_best10 = y
# create and fit model
LogReg_best10= LogisticRegression(solver='newton-cg')
LogReg_best10.fit(x_train_best10, y_train_best10)


# In[68]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#make predictions about what our dependent values would be for training set(yes or no)
y_train_pred10 = LogReg_best10.predict(x_train_best10)
#make a confusion matrix to see results of how our model performs 
confusion_matrix = confusion_matrix(y_train, y_train_pred10)
print(classification_report(y_train, y_train_pred10))


# In[31]:


x_test_best = merged2[best_features]
predictions_best10 = LogReg_best10.predict(x_test_best)
predictions_best10


# In[88]:


print("Precision: " + str(precision_score(y_train_pred10, y_train, average='micro')))
print("Recall: " + str(recall_score(y_train_pred10, y_train, average='micro')))
print("F-1 Training Score: " + str(f1_score(y_train_pred10, y_train, average='micro')))

print("F-1 Test Score based on results of submission: " + str(0.88547) )


# ## Decison Tree

# In[22]:


#using decison tree
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets


# In[102]:


# model
tree = DecisionTreeClassifier()
# train
tree.fit(x_train, y_train)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#make predictions about what our dependent values would be for training set(yes or no)
y_train_pred_dtree = tree.predict(x_train)
#make a confusion matrix to see results of how our model performs 
confusion_matrix = confusion_matrix(y_train, y_train_pred_dtree)
print(classification_report(y_train, y_train_pred_dtree))


# This is probably due to overfitting

# In[103]:


predictions_tree = tree.predict(x_test)
#Score returns the mean accuracy on the given test data and labels
print(tree.score(x_train, y_train))
predictions_dtree = tree.predict(x_test)
predictions_dtree


# In[106]:


print("Precision: " + str(precision_score(y_train_pred_dtree, y_train, average='micro')))
print("Recall: " + str(recall_score(y_train_pred_dtree, y_train, average='micro')))
print("F-1 Training Score: " + str(f1_score(y_train_pred_dtree, y_train, average='micro')))

print("F-1 Test Score based on results of submission: " + str(0.88668) )


# ## Using Support Vector Machines (Unfininshed, wont run in reasonable time)

# In[ ]:


from sklearn.svm import SVC
from sklearn.linear_model import  LogisticRegression
#classifier = SVC(kernel="linear")
#classifier.fit(x_train_best10,np.ravel(y_train_best10))


# ## Random Forest

# In[87]:


# model
forest = RandomForestClassifier(criterion = 'entropy', random_state = 42)

# train
forest.fit(x_train, y_train)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#make predictions about what our dependent values would be for training set(yes or no)
y_train_pred_RFtree = forest.predict(x_train)
#make a confusion matrix to see results of how our model performs 
confusion_matrix = confusion_matrix(y_train, y_train_pred_RFtree)
print(classification_report(y_train, y_train_pred_RFtree))


# In[107]:


print("Precision: " + str(precision_score(y_train_pred_RFtree, y_train, average='micro')))
print("Recall: " + str(recall_score(y_train_pred_RFtree, y_train, average='micro')))
print("F-1 Training Score: " + str(f1_score(y_train_pred_RFtree, y_train, average='micro')))

print("F-1 Test Score based on results of submission: " + str(0.8712) )


# ## Submission ##

# In[ ]:


#can use predictions_best10, predictions_dtree, ect.
submission = pd.concat([merged2.id, pd.Series(predictions)], axis = 1)
submission.columns = ['id', 'Predicted']
submission.to_csv("submission.csv")
#files.download("submission.csv")


# In[ ]:





# In[ ]:





# In[ ]:




