#!/usr/bin/env python
# coding: utf-8

# The data and the description:
# https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks
# 
# Abstract: The datasets' positive class consists of component failures for a specific component of the APS system. The negative class consists of trucks with failures for components not related to the APS.
# 

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
pd.options.display.max_columns = None

import warnings
warnings.filterwarnings('ignore')


# In[3]:


df_original = pd.read_csv('../input/aps_failure_training_set_processed_8bit.csv', dtype = 'str')
df_original = df_original.replace(r'na', np.nan, regex=True)


# In[4]:


df_original.head()


# In[5]:


#encode labels to 0 and 1
le = LabelEncoder()
df_original['class'] = le.fit_transform(df_original['class'])
df = df_original.copy()
df.head()


# The set is very unbalanced with one label (0) being more frequent than the other (1). The algorithm needs to adjust for that. 
# It is done using 'class_weight' hyperparameter which is the ratio of number of 0s to 1s in the label.

# In[6]:


df = df_original.copy()
from sklearn.model_selection import train_test_split
X, y = df.iloc[:,1:], df.iloc[:,0]
X_tr, X_t, y_tr, y_t = train_test_split(X, y, test_size = 0.2, random_state = 0)

weight = sum(y_tr == 0)/sum(y_tr == 1)
lr_full = LogisticRegression(C = 1, class_weight={1:weight}, random_state = 0)
lr_full.fit(X_tr, y_tr)
y_pred = lr_full.predict(X_t)

#calculate the score using confusion matrix values
def score(cm):
    cm_score = cm[0][1] * 10 + cm[1][0] * 500
    cm_score = int(cm_score * 1.33) #1.33 is because the actual test set is 33% larger than this test set
    return cm_score
#calculate confusion matrix
cm = confusion_matrix(y_t, y_pred)
score(cm)


# 13632 is our basic score. We'll use it as a reference for further optimizations.
# The data seemed to be scaled but let's apply scaling to the data just in case. 

# In[7]:


#testing scaling
df = df_original.copy()

from sklearn.preprocessing import MinMaxScaler
scaler_minmax = MinMaxScaler()
X, y = df.iloc[:,1:], df.iloc[:,0]
X_scaled = scaler_minmax.fit_transform(X.values)
X_tr, X_t, y_tr, y_t = train_test_split(X_scaled, y, test_size = 0.2, random_state = 0)

weight = sum(y_tr == 0)/sum(y_tr == 1)
lr_full = LogisticRegression(C = 1, class_weight={1:weight}, random_state = 0)
lr_full.fit(X_tr, y_tr)
y_pred = lr_full.predict(X_t)

#calculate confusion matrix
cm = confusion_matrix(y_t, y_pred)
score(cm)


# MinMaxScaler slightly improved the result. 
# 
# Let's tune the 'C', a hyperparameter (parameter) of the Logistic Regression algorithm.

# In[8]:


#tuning hyperparameters for Logistic Regression
df = df_original.copy()
X, y = df.iloc[:,1:], df.iloc[:,0]
X_tr, X_t, y_tr, y_t = train_test_split(X, y, test_size = 0.2, random_state = 0)
Cs = list(np.arange(0.1, 0.5, 0.1))
weight = sum(y_tr == 0)/sum(y_tr == 1)
for C_ in Cs:  
    lr_full = LogisticRegression(C = C_, class_weight={1:weight}, random_state = 0)
    lr_full.fit(X_tr, y_tr)
    y_pred = lr_full.predict(X_t)

    #calculate confusion matrix
    cm = confusion_matrix(y_t, y_pred)
    score(cm)
    print("C is {0}. Score is: {1}".format(C_, score(cm)))


# C = 0.1 gives the best score.

# Let's try another algorithm. Maybe Random Forest will perfom better.

# In[9]:


#check algorithm with all NAs replaced with mean column values (none rows/columns dropped)
df = df_original.copy()
X, y = df.iloc[:,1:], df.iloc[:,0]
#split into train and test
from sklearn.model_selection import train_test_split
X_tr, X_t, y_tr, y_t = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 200, oob_score = True, class_weight={1:weight}, random_state = 0, bootstrap = True)
rf.fit(X_tr, y_tr)
y_pred = rf.predict(X_t)
cm = confusion_matrix(y_t, y_pred)
score(cm)


# 61938 is significanlty worse.

# It seems Logistic Regression gives us the best score.
# We need to train it on the full training data set and fit the actual test set to get the final score.

# In[10]:


df = df_original.copy()

X_train, y_train = df.iloc[:,1:], df.iloc[:,0]
X_train_scaled = scaler_minmax.fit_transform(X_train.values)
#calculation of the score for the actual test set
weight = sum(y_train == 0)/sum(y_train == 1)
log_reg = LogisticRegression(class_weight = {1:weight}, C = 0.2, random_state=1)
log_reg.fit(X_train_scaled, y_train)

#process the test data set
df_test = pd.read_csv('../input/aps_failure_test_set_processed_8bit.csv', dtype = 'str')
df_test = df_test.replace(r'na', np.nan, regex=True)
    
le = LabelEncoder()
df_test['class'] = le.fit_transform(df_test['class'])
X_test, y_test = df_test.iloc[:,1:], df_test.iloc[:,0]

X_test_scaled = scaler_minmax.transform(X_test.values)
#predict the class for the test set
y_test_pred = log_reg.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_test_pred)
def score(cm):
    cm_score = cm[0][1] * 10 + cm[1][0] * 500
    cm_score = int(cm_score)
    return cm_score
score(cm)


# The final score is 14520. It is not the best score, but a good one. Scheduling repairs according to the algorithm's predictions will significantly reduce the cost of truck repairs.
