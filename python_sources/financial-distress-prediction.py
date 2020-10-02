#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score


# I have modeified the target values if "Financial Distress" if it is greater than -0.50 the company should be considered as healthy (0). Otherwise, it would be regarded as financially distressed (1). 

# In[3]:


data=pd.read_csv('../input/dataset/Financial Distress.csv')


# In[4]:


data.head()


# In[5]:


data.columns


# In[6]:


data['Financial Distress'].value_counts()


# In[7]:


X, y = data.loc[:,data.columns!='Financial Distress'], data.loc[:,'Financial Distress'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=123,stratify=data['Financial Distress'])


# In[8]:


# We have used stratified above to split the data distribution in equal manner
print(pd.value_counts(y_train)/y_train.size * 100)
print(pd.value_counts(y_test)/y_test.size * 100)


# Accuracy is not the best metric to use when evaluating imbalanced datasets as it can be misleading. Metrics that can provide better insight include:
# 
#     Confusion Matrix: a talbe showing correct predictions and types of incorrect predictions.
#     Precision: the number of true positives divided by all positive predictions. Precision is also called Positive Predictive Value. It is a measure of a classifier's exactness. Low precision indicates a high number of false positives.
#     Recall: the number of true positives divided by the number of positive values in the test data. Recall is also called Sensitivity or the True Positive Rate. It is a measure of a classifier's completeness. Low recall indicates a high number of false negatives.
#     F1: Score: the weighted average of precision and recall.
# 
# Since our main objective with the dataset is to prioritize accuraltely classifying financial instability cases the recall score can be considered our main metric to use for evaluating outcomes.

# ## Trying different algorithm

# In[9]:


from sklearn.ensemble import RandomForestClassifier
# train model
rfc = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)

# predict on test set
rfc_pred = rfc.predict(X_test)

accuracy_score(y_test, rfc_pred)


# In[10]:


# f1 score
f1_score(y_test, rfc_pred)


# In[11]:


# confusion matrix
pd.DataFrame(confusion_matrix(y_test, rfc_pred))


# In[12]:


# recall score
recall_score(y_test, rfc_pred)


# Oversampling Minority Class
# 
# Oversampling can be defined as adding more copies of the minority class. Oversampling can be a good choice when you don't have a ton of data to work with. A con to consider when undersampling is that it can cause overfitting and poor generalization to your test set.
# 
# We will use the resampling module from Scikit-Learn to randomly replicate samples from the minority class.
# Important Note
# 
# Always split into test and train sets BEFORE trying any resampling techniques! Oversampling before splitting the data can allow the exact same observations to be present in both the test and train sets! This can allow our model to simply memorize specific data points and cause overfitting.

# In[13]:


from sklearn.utils import resample


# In[14]:


X['class']=y


# In[15]:


X.columns


# In[16]:


# separate minority and majority classes
not_distress = X[X['class']==0]
distress = X[X['class']==1]

# upsample minority
fraud_upsampled = resample(distress,
                          replace=True, # sample with replacement
                          n_samples=len(not_distress), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([not_distress, fraud_upsampled])

# check new class counts
upsampled['class'].value_counts()


# In[17]:


# trying logistic regression again with the balanced dataset
y_train = upsampled['class']
X_train = upsampled.drop('class', axis=1)

upsampled = LogisticRegression(solver='liblinear').fit(X_train, y_train)

upsampled_pred = upsampled.predict(X_test)


# In[18]:


# Checking accuracy
accuracy_score(y_test, upsampled_pred)


# In[19]:


# f1 score
f1_score(y_test, upsampled_pred)


# In[20]:


# confusion matrix
pd.DataFrame(confusion_matrix(y_test, upsampled_pred))


# In[21]:


recall_score(y_test, upsampled_pred)


# Undersampling Majority Class
# 
# Undersampling can be defined as removing some observations of the majority class. Undersampling can be a good choice when you have a ton of data -think millions of rows. But a drawback to undersampling is that we are removing information that may be valuable.
# 
# We will again use the resampling module from Scikit-Learn to randomly remove samples from the majority class.

# In[22]:


# downsample majority
not_fraud_downsampled = resample(not_distress,
                                replace = False, # sample without replacement
                                n_samples = len(distress), # match minority n
                                random_state = 27) # reproducible results

# combine minority and downsampled majority
downsampled = pd.concat([not_fraud_downsampled, distress])

# checking counts
downsampled['class'].value_counts()


# In[23]:


y_train = downsampled['class']
X_train = downsampled.drop('class', axis=1)

undersampled = LogisticRegression(solver='liblinear').fit(X_train, y_train)

undersampled_pred = undersampled.predict(X_test)


# In[24]:


# Checking accuracy
accuracy_score(y_test, undersampled_pred)


# In[25]:


# f1 score
f1_score(y_test, undersampled_pred)


# In[26]:


# confusion matrix
pd.DataFrame(confusion_matrix(y_test, undersampled_pred))


# In[27]:


recall_score(y_test, undersampled_pred)


# Generate Synthetic Samples
# 
# SMOTE or Synthetic Minority Oversampling Technique is a popular algorithm to creates sythetic observations of the minority class.

# In[28]:


from imblearn.over_sampling import SMOTE

# Separate input features and target
y = data['Financial Distress']
X = data.drop('Financial Distress', axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)

sm = SMOTE(random_state=27, ratio=1.0)
X_train, y_train = sm.fit_sample(X_train, y_train)


# In[29]:


smote = LogisticRegression(solver='liblinear').fit(X_train, y_train)

smote_pred = smote.predict(X_test)

# Checking accuracy
accuracy_score(y_test, smote_pred)


# In[30]:


# f1 score
f1_score(y_test, smote_pred, average='weighted')


# In[31]:


# confustion matrix
pd.DataFrame(confusion_matrix(y_test, smote_pred))


# In[32]:


recall_score(y_test, smote_pred)


# Over here undersampling worked the best but many other algorithms like LightGBM, XGBoost will work better which I will try to cover later in this notebook.
