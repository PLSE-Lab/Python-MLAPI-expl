#!/usr/bin/env python
# coding: utf-8

# **Earthquake Prediction Model using Laboratory Data**
# 
# Forecasting earthquakes is one of the most important problems in Earth science because of their devastating consequences. Current scientific studies related to earthquake forecasting focus on three key points: when the event will occur, where it will occur, and how large it will be.
# 
# In this notebook we will work on addressing when the earthquake will take place. 
# *We will try to predict the time remaining before laboratory earthquakes occur from real-time seismic data.*
# 
# Predicting earthquake can save billions of dollars in infrastructure **but above all a large number of human lives can be saved.**

# In[ ]:


import numpy as np 
import pandas as pd 
import os
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from catboost import CatBoostRegressor, Pool
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#print(os.listdir("../input"))


# **1)  Load the Data**

# In[ ]:


train=pd.read_csv("../input/train.csv", nrows=6000000,dtype={'acoustic_data':np.int16,'time_to_failure':np.float64})
train.head(10)


# **2) Exploratory Data Analysis**

# In[ ]:


#Lets plot the data to see and understand the data columns and our problem .
#We will use a small subset of dataset for understanding the pattern ,since the data is large

train_acoustic_df = train['acoustic_data'].values[::100]
train_time_to_failure_df = train['time_to_failure'].values[::100]

fig, ax1 = plt.subplots(figsize=(10,10))
plt.title('Acoustic data and Time to Failure')
plt.plot(train_acoustic_df, color='r')
ax1.set_ylabel('acoustic data', color='r')
plt.legend(['acoustic data'], loc=(0.01, 0.9))

ax2 = ax1.twinx()
plt.plot(train_time_to_failure_df, color='b')
ax2.set_ylabel('time to failure', color='b')
plt.legend(['time to failure'], loc=(0.01, 0.8))

plt.grid(True)

    


# The size of Train data is large . The two columns in the train dataset have the following meaning:
# 
# 1. *accoustic_data*: is the accoustic signal measured in the laboratory experiment;
# 2. *time to failure*: this gives the time until a failure will occurs.
# 
# **The above plot shows that the failure occur after a large oscilation and also that the large oscilation is followed by a series of small minor oscilations before the final time of failure .**

# **3) Feature Engineering**

# In[ ]:


def gen_features(X):
    fe = []
    fe.append(X.mean())
    fe.append(X.std())
    fe.append(X.min())
    fe.append(X.max())
    fe.append(X.kurtosis())
    fe.append(X.skew())
    fe.append(np.quantile(X,0.01))
    fe.append(np.quantile(X,0.05))
    fe.append(np.quantile(X,0.95))
    fe.append(np.quantile(X,0.99))
    fe.append(np.abs(X).max())
    fe.append(np.abs(X).mean())
    fe.append(np.abs(X).std())
    return pd.Series(fe)


# In[ ]:


#Lets read the training set again now in chunks and append features 
train = pd.read_csv('../input/train.csv', iterator=True, chunksize=150_000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

X_train = pd.DataFrame()
y_train = pd.Series()
for df in train:
    ch = gen_features(df['acoustic_data'])
    X_train = X_train.append(ch, ignore_index=True)
    y_train = y_train.append(pd.Series(df['time_to_failure'].values[-1]))


# In[ ]:


X_train.head(10) #Let's check the training dataframe


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id') #Taking the segment id from sample_submission file


# In[ ]:


#Applying Feature Engineering on test data files
X_test = pd.DataFrame()
for seg_id in submission.index:
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    ch = gen_features(seg['acoustic_data'])
    X_test = X_test.append(ch, ignore_index=True)


# In[ ]:


X_test.head(10) #Lets check the testing dataframe


# **4) Catboost Regressor ** *(optional)*

# In[ ]:


#Catboost regressor model 
"""       
#Catboost Regressor model

train_pool = Pool(X_train, y_train)

m = CatBoostRegressor(iterations=10000, loss_function='MAE', boosting_type='Ordered')
m.fit(X_train, y_train, silent=True)
m.best_score_
"""


# **5) Scale the Data**

# In[ ]:


#Scale Train Data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train))
X_train_scaled.head(10)


# In[ ]:


#We will also scale the train data
X_test_scaled = pd.DataFrame(scaler.transform(X_test))
X_test_scaled.head(10)


# **6) Support Vector Regression**

# *We will be using the SVR model for prediction and GridSearchCV for hyperparameter tuning of the model.*

# In[ ]:


parameters = [{'gamma': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
               'C': [0.1, 0.2, 0.25, 0.5, 1, 1.5, 2]}]

reg1 = GridSearchCV(SVR(kernel='rbf', tol=0.01), parameters, cv=5, scoring='neg_mean_absolute_error')


# In[ ]:


reg1.fit(X_train_scaled, y_train.values.flatten())


# In[ ]:


submission.time_to_failure = reg1.predict(X_test_scaled) 
submission


# **Submission**

# In[ ]:


submission.to_csv('submission.csv',index=True)


# If you found this notebook helpful or you just liked it , some upvotes would be very much appreciated - That will keep me motivated .
# 
# Please drop down suggestions and comments if any, so that i can learn to build better solutions.
# 
# **Thank You :-)**

# **Let's Pray that this Year no major earthquake occurs and people remain safe.  **

# 
