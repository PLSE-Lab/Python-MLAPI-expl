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


# In[ ]:


# Data
train = pd.read_csv("/kaggle/input/widsdatathon2020/training_v2.csv")
test = pd.read_csv("/kaggle/input/widsdatathon2020/unlabeled.csv")
# Dictionary
dictionary = pd.read_csv("/kaggle/input/widsdatathon2020/WiDS Datathon 2020 Dictionary.csv")
# Samples
samplesubmission = pd.read_csv("/kaggle/input/widsdatathon2020/samplesubmission.csv")
solution_template = pd.read_csv("/kaggle/input/widsdatathon2020/solution_template.csv")


# ## Baseline Model

# ##### Challengue
# The challenge is to create a model that uses data from the first 24 hours of intensive care to predict patient survival. MIT's GOSSIS community initiative, with privacy certification from the Harvard Privacy Lab, has provided a dataset of more than 130,000 hospital Intensive Care Unit (ICU) visits from patients, spanning a one-year timeframe

# In[ ]:


dictionary


# In[ ]:


dictionary.Category.unique()


# In[ ]:


#dictionary[dictionary['Data Type'] == 'numeric']


# ##### Training data
# Labeled training data are provided for model development; you will then upload your predictions for unlabeled data to Kaggle and these predictions will be used to determine the public leaderboard rankings, as well as the final winners of the competition.****

# In[ ]:


train


# In[ ]:


test


# In[ ]:


train.hospital_death.describe()


# In[ ]:


print('unlabeled data: {}\ntraining data:  {}'.format(test.shape, train.shape))


# ##### Model
# Logistic regression predicts the probability of an outcome that can only have two values (i.e. a dichotomy). The prediction is based on the use of one or several predictors.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[ ]:


train.fillna(method='ffill', inplace=True)
test.fillna(method='ffill', inplace=True)


# In[ ]:


# Train
X_train = train[['age','weight']].values
y_train = train['hospital_death'].values
# Test
X_test = test[['age','weight']].values


# In[ ]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[ ]:


y_predict = model.predict_proba(X_test)


# In[ ]:


test['hospital_death'] = y_predict[:,0]


# In[ ]:


test


# In[ ]:


test[["encounter_id","hospital_death"]].to_csv("submission.csv",index=False)


# In[ ]:


test[["encounter_id","hospital_death"]]


# Tuning the Model
