#!/usr/bin/env python
# coding: utf-8

# # Why should you work with this type of dataset
# If you work on this type of dataset, you can implmenet similar solutions in any IoT related project in your organization or personal project. 
# 
# This dataset can help you to learn:
#     - how to approach sensors data
#     - how to find anomaly which can help you to know when device is going to break
#     - It do have timeseries angle, so could look for that too. 

# # Step 1. 
#  - Have a quick look on the dataset, which is very much needed to build the thought process around the data
#  - Reread problem statement multiple times and try to understand how to correlate the dataset and problem. 
#  - Plot the visualization 
#  
#  Our problem statment: Look for the correlation of sensors which leads to device breakdown
# 

# 
# 1. Quick Checklist for this dataset
# *     timeseries forecasting problem 
# *     machine status - 3 convert that into label encoding 
# *     all numerical values
# *     anomlay detection 
# *     labelled data - supervised learning , classification 
# *     look for correlation matrix 
# *     look for skewness in the data 
# *     check imbalance angle if any 

# In[ ]:


# load the dataset 
import pandas as pd
import numpy as np

df = pd.read_csv('../input/sensor.csv')
df.head(2)


# In[ ]:


df.tail()

# 01-Apr-2018 to 31-Aug-2018 
# apr, may, jun, jul, aug - 5 months every min data is collected 


# In[ ]:


del df['Unnamed: 0']


# In[ ]:


# convert time into index 
df['index'] = pd.to_datetime(df['timestamp'])
df.index = df['index']


# In[ ]:


# delete the colunmns 
del df['index']
del df['timestamp']


# In[ ]:


df.head(2)


# In[ ]:


df['sensor_15'].nunique() # no unique - complete zero
# drop the column 
df.drop(['sensor_15'], axis=1, inplace = True)
df.shape


# In[ ]:


df.info()


# In[ ]:


# machine status - no null 
# we will drop na in whole dataframe 
df['sensor_00'].isna().sum()


# In[ ]:


# machine status
df['machine_status'].unique()#'NORMAL', 'BROKEN', 'RECOVERING' 
df['machine_status'].value_counts()


# In[ ]:


# draw a countplot for machine status 
import seaborn as sns
sns.countplot(y = df['machine_status'])


# ** We are going to figureout what makes device to breakdown, 
# * it is highly imbalanced data  and undersampling can't help here 
# * we will experiement with SMOTE or oversampling 

# In[ ]:


# apply label encoder to encode the machine status
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['machine_status'] = le.fit_transform(df['machine_status'])
df['machine_status'].value_counts()

# 1 - normal 
# 2 - recovering 
# 0 - broken


# In[ ]:


#  look on complete data frame when device is broken
df_broken = df[df.machine_status ==0]
df_broken

# there is no nan value corellation for broken device 
#


# In[ ]:


import matplotlib.pyplot as plt 
plt.plot(df['sensor_02'])


# In[ ]:


# imputation for null values 
df['sensor_04'].hist()
# data is skewwed so we need to use median value to fill the data


# In[ ]:


# let us figureout NaN values 
df['sensor_00'].isna().sum()


# In[ ]:


df['sensor_50'].isna().sum()


# In[ ]:


# used ffill method to fill the missing values
df = df.fillna(method='ffill')


# In[ ]:


X = df.drop(['machine_status'], axis=1)
X.shape


# In[ ]:





# In[ ]:


Y = df['machine_status']
Y.shape


# In[ ]:


# apply the logitic regression 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.30, random_state = 42)


# In[ ]:


# apply 
logit = LogisticRegression()
model = logit.fit(X_train, y_train)


# In[ ]:


# predict
y_pred = model.predict(X_test)


# In[ ]:


# evaluate the model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
cm = pd.crosstab(y_test,y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
cm


# In[ ]:


# accuracy is not a good metrics for Anomaly detection and imblaanced dataset
accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[ ]:


# Classification Report
cr = classification_report(y_pred, y_test)
print(cr)


# # these reports are not good 
# we will use Isolation forest and oneSVM for modelling 
# xgboosting feature_importance and PCA for dimension reduction 
# before that we will divide this dataset into 2 probelms
#  machine status - normal + broken, normal + recovery, recovery+ broken 

# In[ ]:


df.shape # look on the shape of the dataset


#  machine status - normal + broken

# In[ ]:


df1 = df.copy()
df1 = df[(df1.machine_status ==1) | (df1.machine_status ==0)]
df1.shape


# 
