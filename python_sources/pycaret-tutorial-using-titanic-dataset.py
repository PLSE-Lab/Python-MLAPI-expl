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


# # Pip install PyCaret

# In[ ]:


get_ipython().system('pip3 install pycaret')


# # Loading the Required Packages

# In[ ]:


import pandas as pd
from pycaret import classification


# # Getting the Data

# In[ ]:


train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")


# # Setting up Environment

# In[ ]:


classification_setup = classification.setup(data=train,target='Survived')


# *Once the setup has been succesfully executed it prints the information grid which contains several important pieces of information. Most of the information is related to the pre-processing pipeline which is constructed when setup() is executed.*

# # Compare Model

# In[ ]:


classification.compare_models()


# **Note** : *Comparing all models to evaluate performance is the recommended starting point for modeling once the setup is completed (unless you exactly know what kind of model you need, which is often not the case). This function trains all models in the model library and scores them using kfold cross validation for metric evaluation.* 

# In[ ]:


# Create Xgboost model
classification_xgb = classification.create_model('xgboost')


# *The output prints a score grid that shows average Accuracy, AUC, Recall, Precision, F1, Kappa accross the folds (10 by default) of all the available models in the model library.*

# In[ ]:


# Tune the model
tune_xgb = classification.tune_model('xgboost')


# *Obeserve the difference when the model is run with default parameters and after fine tuning them*

# In[ ]:


# build the lightgbm model
classification_lightgbm = classification.create_model('lightgbm')


# In[ ]:


# Tune lightgbm model
tune_lightgbm = classification.tune_model('lightgbm')


# # Plot Model

# In[ ]:


# Residual Plot
classification.plot_model(tune_lightgbm)


# In[ ]:


# Error Plot
classification.plot_model(tune_lightgbm, plot = 'error')


# In[ ]:


# Feature Important plot
classification.plot_model(tune_lightgbm, plot='feature')


# In[ ]:


# Evaluate model
classification.evaluate_model(tune_lightgbm)


# # 1. Make Predictions - Xgboost

# In[ ]:


# read the test data
test_data_classification = pd.read_csv("/kaggle/input/titanic/test.csv")
# make predictions
predictions = classification.predict_model(tune_xgb, data=test_data_classification)
# view the predictions
predictions


# # 2 .Make Predictions - lightgbm

# In[ ]:


# read the test data
test_data_classification = pd.read_csv("/kaggle/input/titanic/test.csv")
# make predictions
predictions = classification.predict_model(tune_lightgbm, data=test_data_classification)
# view the predictions
predictions


# # Thanks for reading.If you like it Kindly Upvote

# Special Thanks to : **Moez Ali** - Data Scientist, Founder & Author of PyCaret & **Analytics Vidhya**
# 
# [https://towardsdatascience.com/announcing-pycaret-an-open-source-low-code-machine-learning-library-in-python-4a1f1aad8d46](http://)
# 
# [https://www.analyticsvidhya.com/blog/2020/05/pycaret-machine-learning-model-seconds/?utm_source=feed&utm_medium=feed-articles&utm_campaign=feed](http://)
