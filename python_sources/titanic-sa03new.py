#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
test  = pd.read_csv('../input/titanic/test.csv')
sub   = pd.read_csv('../input/titanic/gender_submission.csv')


# In[ ]:


train.shape, test.shape


# In[ ]:


get_ipython().system('pip install pycaret')


# In[ ]:


# Import whole classification
from pycaret.classification import *


# In[ ]:


train.head(2)


# In[ ]:


train.info()


# In[ ]:


#  Set up our dataset (preprocessing)

clf = setup(data = train, 
             target = 'Survived',
             numeric_imputation = 'mean',
             categorical_features = ['Sex','Embarked'], 
             ignore_features = ['Name','Ticket','Cabin'],
             silent = True)


# In[ ]:


# Compare the models

compare_models()


# In[ ]:


# let's create a Light GBM Model

lgbm  = create_model('lightgbm') 


# In[ ]:


#  Let's tune it!

tuned_lightgbm = tune_model('lightgbm')


# In[ ]:


# Learning Curve

plot_model(estimator = tuned_lightgbm, plot = 'learning')


# In[ ]:


train.shape, test.shape


# In[ ]:


#  AUC Curve

plot_model(estimator = tuned_lightgbm, plot = 'auc')


# In[ ]:


# Confusion Matrix

plot_model(estimator = tuned_lightgbm, plot = 'confusion_matrix')


# In[ ]:


#  Feature Importance

plot_model(estimator = tuned_lightgbm, plot = 'feature')


# In[ ]:


#  whole thing TOGETHER

evaluate_model(tuned_lightgbm)


# In[ ]:


# MODEL Interpretation

interpret_model(tuned_lightgbm)


# In[ ]:


# MODEL Predictions

predict_model(tuned_lightgbm, data=test)


# In[ ]:


predictions = predict_model(tuned_lightgbm, data=test)
predictions


# In[ ]:


sub['Survived'] = round(predictions['Score']).astype(int)
sub.to_csv('submission.csv',index=False)


# In[ ]:


sub

