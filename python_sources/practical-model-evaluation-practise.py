#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, confusion_matrix
import pandas as pd
import category_encoders as ce
import random

random.seed(42)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


## Reading the 2019 data,

data = pd.read_csv("../input/data-prep-for-job-title-classification/data_jobs_info_2019.csv")

data.head()


# In[ ]:


## Split data into train and test sets:

X=data.drop("job_title",axis=1)
Y=data['job_title']


# In[ ]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.80)


# In[ ]:


## Saving our data to use in Cloud Auto ML:

with open("train_data.csv","+w") as f:
    pd.concat([X_train,Y_train],axis=1).to_csv(f,index=False)
    
with open('test_data.csv','+w') as f:
    pd.concat([X_test,Y_test],axis=1).to_csv(f,index=False)


# In[ ]:


## Doing ordinal encoding for TPOT and Xgboost:

encoding_x=ce.OrdinalEncoder()
X_encoder=encoding_x.fit_transform(X)

encoding_y=ce.OrdinalEncoder()
Y_encoder=encoding_y.fit_transform(Y)


# In[ ]:


X_encoder.head()


# In[ ]:


Y_encoder.head()


# In[ ]:


X_train_encoder,X_test_encoder,Y_train_encoder,Y_test_encoder=train_test_split(X_encoder,Y_encoder,train_size=0.8)


# In[ ]:


### Using TPOT algorithm:

from tpot import TPOTClassifier

# create & fit TPOT classifier with 
tpot = TPOTClassifier(generations=10, population_size=20, 
                      verbosity=2, early_stop=2,cv=5)
tpot.fit(X_train_encoder,Y_train_encoder)

# save our model code
tpot.export('tpot_pipeline.py')

# print the model code to see what it says
get_ipython().system('cat tpot_pipeline.py')


# In[ ]:


import h2o
from h2o.automl import H2OAutoML

h2o.init()


# In[ ]:


train_data=h2o.H2OFrame(X_train)
y_train_data=h2o.H2OFrame(list(Y_train))

train_data=train_data.cbind(y_train_data)

train_data.head()


# In[ ]:


h2omodel=H2OAutoML(max_models=20,seed=1)

h2omodel.train(y="C1",training_frame=train_data)


# In[ ]:


lb=h2omodel.leaderboard

lb.head(10)


# In[ ]:


h2o.save_model(h2omodel.leader)


# In[ ]:


get_ipython().system('ls')


# In[ ]:




