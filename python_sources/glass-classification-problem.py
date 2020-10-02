#!/usr/bin/env python
# coding: utf-8

# **Glass Classification with Machine Learning**
# 
# In this kernal, you are going to experience Glass Classification with Polynomial and Logistic Regression. I am going to present the feature importance and feature dependency for reference and evaluated resulta for the predicted models.
# 
# **Contents**
# 1. Load Libraries and Ready the dataset for training and testing.
# 2. Predicting the feature importance.
# 3. Feature dependency (i.e., HeatMap).
# 4. Value Prediction.
# 
# **1. Load Libraries and Split the DataSet**.
# 
#    Let me load the libraries that are to be used in this kernal.
# 
# 

# In[3]:


# Loading the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Loading the DataSet
df = pd.read_csv('../input/glass.csv')

X = df[['Type']]
y = df.drop('Type',1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# **2. Predicting **
# 
# Predicting the features is done by Random Forest Classifier and the results are displayed.

# In[5]:


### Feature Importance

rf = RandomForestClassifier()
rf.fit(y.values, X.values.ravel())

importance = rf.feature_importances_
importance = pd.DataFrame(importance, index = y.columns, columns=['Importance'])

feats = {}
for feature, importance in zip(y.columns,rf.feature_importances_):
    feats[feature] = importance
    
print (feats)
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance').plot(kind='bar', rot=45)


# **3. Feature Dependency**
# 
# Feature dependency is caluculated using seaborn and heatmap.

# In[6]:


### HeatMap

y_cols = y.columns.tolist()
corr = df[y_cols].corr()

sns.heatmap(corr)


# **4. Value Prediction**
# 
# The test values are predicted using Polynomial and Logistic Regression. The Mean Squared Error and Confusion Matrix are provided to self evaluate the predicted values. They are as follows:

# In[7]:


### Prediction

model = PolynomialFeatures(degree= 4)
y_ = model.fit_transform(y)
y_test_ = model.fit_transform(y_test)

lg = LinearRegression()
lg.fit(y_,X)
predicted_data = lg.predict(y_test_)
predicted_data = np.round_(predicted_data)

print ('Mean Square Error:')
print (mean_squared_error(predicted_data, X_test))
print ('Predicted Values:')
print (predicted_data.ravel())

### Correlation Matrix
print ('')
print ('Confusion Matix:')
print (confusion_matrix(X_test, predicted_data))


# In[ ]:




