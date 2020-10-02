#!/usr/bin/env python
# coding: utf-8

# # Predictive analysis of Bank Marketing
# 
# #### Problem Statement
# The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 
# 
# #### What to achieve?
# The classification goal is to predict if the client will subscribe a term deposit (variable y).
# 
# #### Data Contains information in following format:
# 
# ### Categorical Variable :
# 
# * Marital - (Married , Single , Divorced)",
# * Job - (Management,BlueCollar,Technician,entrepreneur,retired,admin.,services,selfemployed,housemaid,student,unemployed,unknown)
# * Contact - (Telephone,Cellular,Unknown)
# * Education - (Primary,Secondary,Tertiary,Unknown)
# * Month - (Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec)
# * Poutcome - (Success,Failure,Other,Unknown)
# * Housing - (Yes/No)
# * Loan - (Yes/No)
# * Default - (Yes/No)
# 
# ### Numerical Variable:
# 
# * Age
# * Balance
# * Day
# * Duration
# * Campaign
# * Pdays
# * Previous
# 
# #### Class
# * deposit - (Yes/No)

# In[ ]:


#Importing required libraries
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import statsmodels.formula.api as smf
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Importing and displaying data
data = pd.read_csv("../input/bank.csv", delimiter=";",header='infer')
data.head()


# In[ ]:


#4521 rows and 17 features
data.shape


# In[ ]:


#datatypes of the columns
data.dtypes


# Since the dtype contains types other than int, floot; we need to convert those column values into proper format in order to fit the data in model.

# In[ ]:


#Converting object type data into numeric type using One-Hot encoding method which is
#majorly used for XGBoost (for better accuracy) [Applicable only for non numeric categorical features]
data_new = pd.get_dummies(data, columns=['job','marital',
                                         'education','default',
                                         'housing','loan',
                                         'contact','month',
                                         'poutcome'])
#pd is instance of pandas. Using get_dummies method we can directly convert any type of data into One-Hot encoded format.


# In[ ]:


#Since y is a class variable we will have to convert it into binary format. (Since 2 unique class values)
data_new.y.replace(('yes', 'no'), (1, 0), inplace=True)


# In[ ]:


#Checking types of all the columns converted
data_new.dtypes


# In[ ]:


#Our New dataframe ready for XGBoost
data_new.head()


# In[ ]:


#Spliting data as X -> features and y -> class variable
data_y = pd.DataFrame(data_new['y'])
data_X = data_new.drop(['y'], axis=1)
print(data_X.columns)
print(data_y.columns)


# In[ ]:


#Dividing records in training and testing sets along with its shape (rows, cols)
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=2, stratify=data_y)
print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)


# In[ ]:


#Create an XGB classifier and train it on 70% of the data set.
from sklearn import svm
from xgboost import XGBClassifier
clf = XGBClassifier()
clf


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:


#classification accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))


# **Using xgb Library**

# In[ ]:


import xgboost as xgb
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=2, stratify=data_y)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)


# In[ ]:


watchlist = [(dtrain, 'train'),(dtest, 'val')]
print(watchlist)


# In[ ]:


#Train the model
params = {
    'objective':'multi:softprob',
    'max_dept':4,
    'silent':1,
    'eta':0.3,
    'gamma': 0,
    'num_class': 2
}
num_rounds=20


# In[ ]:


XGB_Model = xgb.train(params,dtrain,num_rounds)


# In[ ]:


XGB_Model.dump_model('dump.rawBank.txt')


# In[ ]:


y_predict = XGB_Model.predict(dtest)
#print(y_predict)


# In[ ]:


from xgboost import plot_importance
from matplotlib import pyplot
plot_importance(XGB_Model)
pyplot.show()


# In[ ]:


#Tree visualisation (Double tap to zoo)
xgb.plot_tree(XGB_Model, num_trees=2)
fig = plt.gcf()
fig.set_size_inches(150, 100)
fig.savefig('tree.png')


# In[ ]:




