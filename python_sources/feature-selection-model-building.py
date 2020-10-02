#!/usr/bin/env python
# coding: utf-8

# 
# # Telecom customer churn dataset:Feature selection and Model building

# #### Previously to this notebook, Data Analysis and Data feature engineering activities where executed.
# 
# For a description of the dataset, please, visit [Kaggle.com] https://www.kaggle.com/abhinav89/telecom-customer
# 
# In a previous notebook, feature engineering was performed. The following was checked:
#  1. Checking if the dataset was balanced.
#  2. Categorical variables. Rare labels, encoding, missing values.
#  3. Continuos numerical variables. Missing values,  Outliers, distribution.
#  4. Discrete numerical variables. Missing values. 
#  5. Feature Scaling 
# 
# In this notebook the following will be developed:
#  1. Feature selection. In order to get the best variables for a Logistic model. 
#  2. A Logistic model was built given 58% of accuracy.  
#  3. A radomForest algorithm. In general, it performs better than the Logictic Regression. Better score, precision and recall. 

# In[1]:


# to handle datasets
import pandas as pd
import numpy as np
import os

# for plotting
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sn

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)


# In[2]:


# load dataset
# We  load the datasets with the engineered values
# Reading the engineered training and testing telecom cunsoters' dataset
data = pd.read_csv('../input/data_fe.csv')
print(data.shape)
data.head()


# In[3]:


# capture the variables and target datasets
Id = 'Customer_ID'
target = 'churn'
x = data.iloc[:, data.columns != target]
x = x.iloc[:, x.columns != Id]
y = data.iloc[:, data.columns == target]


# ### Feature Selection
# 
# Let's go ahead and select a subset of the most predictive features. 

# In[4]:


from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn import metrics
from math import sqrt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics


# In[20]:


# creating the model
lmodel = LogisticRegression(solver = 'liblinear', random_state=1,C=1.0, max_iter = 10)


# In[21]:


# Getting selected features
sel_f = RFE(lmodel,30)
#sel_f = SelectFromModel(lmodel)


# In[22]:


# fiting the model
sel_f = sel_f.fit(x, y.values.ravel())


# In[23]:


sel_f.get_support()


# In[24]:


x.columns[sel_f.get_support()]


# In[25]:


# list of the selected features
#selected_feat = x.columns[(sel_f.support_)]
selected_feat = x.columns[sel_f.get_support()]

# printing number of features
print('total features: {}'.format((x.shape[1])))
print('selected features: {}'.format(len(selected_feat)))


# In[26]:


# Getting the train and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x[selected_feat], y, test_size = 0.25, random_state = 1)


# In[27]:


# fiting the model
lmodel.fit(x_train,np.array(y_train).ravel())


# In[28]:


list(zip(x.columns[sel_f.get_support()],np.transpose(lmodel.coef_)))


# In[29]:


# Getting the scores
print("R2:",lmodel.score(x_train,np.array(y_train).ravel()))


# In[30]:


y_pred_test = lmodel.predict(x_test)


# ### Cross Validation

# In[31]:


scores = cross_val_score(lmodel, x, np.array(y).ravel(), scoring="accuracy", cv=20)
print("cros Validation score mean:",scores.mean())


# * The cross validation won't improved the score.  

# ### Classification Report

# In[32]:


print(metrics.classification_report(y_test, y_pred_test, target_names=["N","Y"]))


# ### Confusion matrix

# In[33]:


# building the confusion matrix
confusion_matrix = metrics.confusion_matrix(np.array(y_test), y_pred_test)
plt.figure(figsize = (10,7))
sn.heatmap(confusion_matrix, annot=True)


# ### ROC curve

# In[35]:


# predict probabilities
probs = lmodel.predict_proba(x_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()


# * The ROC curve does not show a good performance either since it is near to flat

# ## RandomForest Algorithm

# In[36]:


from sklearn.ensemble import RandomForestClassifier


# In[37]:


forest = RandomForestClassifier(n_jobs=4, oob_score=True, n_estimators=100,min_samples_leaf=20,min_samples_split=40,
                                random_state =1,criterion='entropy')


# In[38]:


forest.fit(x,y.values.ravel())


# In[39]:


forest.oob_score_


# In[40]:


rfpred = forest.predict(x)


# In[41]:


print(metrics.classification_report(y, rfpred, target_names=["N","Y"]))


# In[42]:


# Building the confusion matrix
confusion_matrix = metrics.confusion_matrix(np.array(y), rfpred)
plt.figure(figsize = (10,7))
sn.heatmap(confusion_matrix, annot=True)


# In[77]:


list(zip(x.columns,np.transpose(forest.feature_importances_)))


# * In general, the RandomForest performs better than the Logictic Regression. 
# * For what can be observered from the features selected, the monthly rent, the number of subscribers in the household, the time using the cellphone, monthly use of the service, roaming service,  change in service use respective to the previous 3 months, time being a client, and service failures, are the variables showing more importance in the behaviour of the client in order to determine churn.

# In[ ]:




