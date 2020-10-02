#!/usr/bin/env python
# coding: utf-8

# <h1>The Blood Transfusion Service Center Dataset</h1>

# <p style="padding:5px;">Given our mission, we're interested in predicting if a blood donor will donate within a given time window.</p>
# 
# <p style="padding-left:6px">Predict if the donor will give in March 2007
# The goal is to predict the last column, whether he/she donated blood in March 2007.</p>
# 
# <p style="padding-left:10px">Use information about each donor's history</p>
# <ul>
#     <li><strong>Months since Last Donation :</strong> this is the number of monthis since this donor's most recent donation.</li>
#     <li><strong>Number of Donations :</strong> this is the total number of donations that the donor has made.</li>
#     <li><strong>Total Volume Donated :</strong> this is the total amound of blood that the donor has donated in cubuc centimeters.</li>
# <li><strong>Months since First Donation:</strong> this is the number of months since the donor's first donation.</li>
# </ul>

# <h3 style="padding-left:10px">Importing Necessary Libraries</h3>

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# <h3 style="padding-left:10px">Reading the dataset</h3>

# <p style="padding-left:10px">We have two files <b>blood-train.csv</b> and <b>blood-test.csv</b></p>
# <ul><li><b>blood-train.csv</b> : Training data</li>
#     <li><b>blood-test.csv</b> : Test data</li></ul>

# In[ ]:


# Train data
df_train = pd.read_csv("../input/blood-train.csv")
df_train.head()


# In[ ]:


# Test data
df_test = pd.read_csv("../input/blood-test.csv")
df_test.head()


# <p>So this is how our data looks like, you can see we have multiple columns in our data but the Unnamed:0 should be either removed or labeled.</p>

# In[ ]:


#labelling
df_train.rename(columns={"Unnamed: 0":"Donor_id"},inplace=True)
df_train.head()


# In[ ]:


df_test.rename(columns={"Unnamed: 0":"Donor_id"},inplace=True)
df_test.head()


# <p>We will explore our data now.</p>

# In[ ]:


df_train.shape, df_test.shape


# In[ ]:


df_train.info()
print("\n--------------------------------------\n")
df_test.info()


# <b>No missing values</b>

# In[ ]:


#Statistical Inference

df_train.describe()


# In[ ]:


df_test.describe()


# In[ ]:


# Correlation
train_corr = df_train.corr()
sns.heatmap(train_corr)


# In[ ]:


test_corr = df_test.corr()
sns.heatmap(test_corr)


# <h1>Data Preprocessing</h1>

# In[ ]:


# Training data
X_train = df_train.iloc[:,[1,2,3,4]].values
y_train = df_train.iloc[:,-1].values


# In[ ]:


X_train,y_train


# In[ ]:


# Test data
X_test = df_test.iloc[:,[1,2,3,4]].values


# In[ ]:


X_test


# In[ ]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)

X_test = Scaler.fit_transform(X_test)


# In[ ]:


X_train, X_test


# <h1>Logistic Regression</h1>
# <p style="padding-left:10px">Building a Logistic Regression model</p>

# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


score = classifier.score(X_train,y_train)
score


# In[ ]:


#Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=5)
mean = accuracies.mean()
std = accuracies.std()


# In[ ]:


mean,std


# <h1>Random Forest</h1>

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train,y_train)
score = rf.score(X_train,y_train)
score


# In[ ]:


y_pred = rf.predict(X_test)


# <h1>XGBoost</h1>

# In[ ]:


from xgboost import XGBClassifier
xg = XGBClassifier()
xg.fit(X_train,y_train)
score = xg.score(X_train,y_train)
score


# In[ ]:


#Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=xg,X=X_train,y=y_train,cv=10)
mean = accuracies.mean()
std = accuracies.std()


# In[ ]:


mean,std


# In[ ]:


y_pred = xg.predict(X_test)

