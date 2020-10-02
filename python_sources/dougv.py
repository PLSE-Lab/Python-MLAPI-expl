#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Predictive Maintenance Workbook and Explanation
# ### Doug Valentine - DS Petroleum Eng

# ## First Import Packages and Import the Data

# In[ ]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


# In[ ]:


df = pd.read_csv('/kaggle/input/equipfailstest/equip_failures_training_set.csv')


# ## Exploratory Analysis And Preparing the Dataset

# In[ ]:


df.describe()


# ### Only 1 column is in numeric dtype
# This next step goes through and turns the string columns into numeric.  Everytime the to_numeric fails, I force the function to input an NaN to take care of later

# In[ ]:


cols = df.columns
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# We need to hand the NaN in the dataset.  If I was using a timeseries type dataset I would use fillna with a forwardfill or backfill.  However, I don't believe this dataset was in order so I will input a placeholder number.  -1 is usuallly a good number for this.

# In[ ]:


df.isna().sum().sum()


# In[ ]:


df.fillna(-1, inplace=True)


# In[ ]:


df.isna().sum().sum()


# ## Balancing the dataset
# A common problem with this type of dataset is having more 'normal' operating conditions than failures.  The models tend to bias guessing 'normal' therefore cheating the system.  This results in low F-1 Scores.  I chose to take all the failure data then randomly sampling only a portion of target == 0 data.  Through experimentation I landed on 1:5 ratio

# In[ ]:


df.target.plot(kind='hist')


# In[ ]:


split=0.2
df_fail = df[df.target == 1]
df_normal = df[df.target == 0].sample(n=int(df_fail.shape[0]/split), random_state=42)
bal = df_fail.append(df_normal)


# In[ ]:


bal.target.plot(kind='hist')


# ## The Machine Learning
# Splitting Test and Training Data.  Also bringing the full dataset through the pipeline as total to score the model against it

# In[ ]:


X, y = bal.iloc[:,2:] , bal['target']
total, test = df.iloc[:,2:] , df['target']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# ### Scaling the data
# Without scaling the data, larger numeric columns can be more impactful.  Standard scaler normalizes the data under a normal distribution

# In[ ]:


scaler = StandardScaler()
scaler.fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
total = scaler.transform(total)


# ### Setting up the gridsearchcv
# I only ran two separate algorithms for time constarints.  I ran the KNearestNeighbors and the RandomForestClassifier.  I will only show the KnearestNeighbors here. If I had more time I would run SVM and likely a full ANN with keras
# 
# It's important to run a gridsearch here to find the optimal hyperparameters.  Also the cross_validation (cv) helps generalize the model and can reduce the risk of overfitting.  Also changing the scoring parameter to F1 as stated in the rules to have the gridsearch utilize f1 and not accuracy score

# In[ ]:


params_grid = {
    'n_neighbors' : list(range(2,20,1)),
    'weights' : ['uniform', 'distance']
}


# In[ ]:


clf = GridSearchCV(KNeighborsClassifier(), params_grid, scoring='f1', cv=5)


# In[ ]:


clf.fit(X_train,y_train)


# Checking the best parameters

# In[ ]:


clf.best_params_


# In[ ]:


y_pred = clf.predict(X_test)


# ### Checking both the F-1 and accuracy scores

# In[ ]:


print(f'The accuracy score: {accuracy_score(y_test, y_pred)}')
print(f'The F1 Score is: {f1_score(y_test,y_pred)}')


# ### Finally taking all the data (prior to balancing the data) to assess if the model fits the data generally.

# In[ ]:


pred = clf.predict(total)


# In[ ]:


print(f'The total accuracy score: {accuracy_score(test, pred)}')
print(f'The total F1 Score is: {f1_score(test,pred)}')


# ## Lastly Productionizing the Model
# Joblib is a good way to store the models and scalers for production.  If I had more time I would put the scaler and model into a sklearn pipeline for further automating the workflow

# In[ ]:


from joblib import dump, load


# In[ ]:


dump(clf, 'gridsearch_knn.joblib')
dump(scaler, 'scaler.joblib')

