#!/usr/bin/env python
# coding: utf-8

# **Credit Card Fraud Analysis and Prediction**
# 
# This notebook will help you to predict the credict card frauds. Feature dependecny and Feature Heatmap are used to show the dependency of target variable on the features and feature self-dependency. I would like to thank Currie32 and credit his notebook named "Predicting Fraud with TensorFlow". The proceedings follows..
# 
# **Contents**
# 1.  Introduction
# 2. Understanding the Data
# 3. Fraud vs Normal Comparisons
# 4. Feature Importance
# 5. Feature HeatMap
# 6. Model and Predictions

# **1. Introduction**
# 
# Load the modules required for caluculation feature importance, feature heatmap and predicting the data. Load the data and look at the shape and colums. This gives an idea what the dataset is really about and it's basic contents.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

df = pd.read_csv('../input/creditcard.csv')
print (df.shape)
df.columns


# **2. Understanding the Data**
# 
# Choose the target variable and feature variables. Split the data into test and train data. This helps to separate a chunck of data and caluculate the accuracy of the prediction model.

# In[2]:


X = df[['Class']]
y = df.drop('Class',1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
df.describe()


# Look into the data separatly, I mean Fraud data and Normal data. This makes us to understand the computational differences in the data.

# In[3]:


print ("Fraud")
print (df.Amount[df.Class == 1].describe())
print ()
print ("Normal")
print (df.Amount[df.Class == 0].describe())


# **3. Fraud Data vs Normal Data**
# 
# The below pictures represnts the differences between Fraud and Normal Data with reference to few feature variables. While looking at *Time* vs *Number of Transactions* we can visualize that Normal transactions follow a cyclical pattern while Fraud transactions doesn't. This might help us in prediction model. Let's move to other feature combinations.

# In[4]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20,6))

bins = 100

ax1.hist(df.Time[df.Class == 1], bins = bins)
ax1.set_title('Fraud')

ax2.hist(df.Time[df.Class == 0], bins = bins)
ax2.set_title('Normal')

plt.xlabel('Time (in Seconds)')
plt.ylabel('Number of Transactions')
plt.show()


# Take a look at the *Amount* vs *Time* graph. The data outliers are too many in fraud transactions where as in normal transactions they are not more than 5.

# In[5]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20,6))

ax1.scatter(df.Time[df.Class == 1], df.Amount[df.Class == 1])
ax1.set_title('Fraud')

ax2.scatter(df.Time[df.Class == 0], df.Amount[df.Class == 0])
ax2.set_title('Normal')

plt.ylabel('Amount ($)')
plt.xlabel('Time')
plt.show()


# **4. Feature Importance**
# 
# * Feature Importance is calculated using Random Forest Classifier.

# In[6]:


rf = RandomForestClassifier()
rf.fit(y.values, X.values.ravel())

importance = rf.feature_importances_
importance = pd.DataFrame(importance, index = y.columns, columns = ['Importance'])

feats = {}
for feature, importance in zip(y.columns,rf.feature_importances_):
    feats[feature] = importance
    
print (feats)
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance').plot(kind='bar', rot=90)


# **5. Feature HeatMap**
# 
# Correlation of features on each other can be observed using seaborn. It could be observed that most of the features are independent of others making it easy to predict.

# In[7]:


plt.subplots(figsize=(15,10))
y_cols = y.columns.tolist()
corr = df[y_cols].corr()

sns.heatmap(corr)


# **6. Model and Prediction**
# 
# Random Forest Classifiers are used to classify the fraud and normal detectors

# In[8]:


rf = RandomForestClassifier(n_estimators = 20)
rf.fit(y_train, X_train)


# The results are predicted and accuracy percentage is displayed. Confusion Matrix is used to give the no. of perfomance issues.

# In[9]:


predicted_data = rf.predict(y_test)
print (rf.score(y_test, X_test))
confusion_matrix(predicted_data, X_test)

