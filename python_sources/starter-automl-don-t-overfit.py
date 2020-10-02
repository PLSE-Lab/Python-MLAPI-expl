#!/usr/bin/env python
# coding: utf-8

# # **Starter H2O for Don't Overfit**
# 
# ---
# 
# ## ***Outline of the notebook***
# 
# ---
# 
# * Step [**1.Import h2o**](#1.Import-h2o)
# * Step [**2.Load Data**](#2.Load-Data)
# * Step [**3.Statistics of Data**](#3.Statistics-of-Data)
# * Step [**4.Split Train and target**](#4.Split-Train-and-target)
# * Step [**5.Automl Model Training**](#5.Automl-Model-Training)
# * Step [**6.Leader Board of Models**](#6.Leader-Board-of-Models)
# * Step [**7.Ensemble Exploration**](#7.Ensemble-Exploration)
# * Step [**8.Model Importance**](#8.Model-Importance)
# * Step [**9.Save-Leader-Model**](#9.Save-Leader-Model)
# * Step [**10.Read test data**](#10.Read-test-data)
# * Step [**11.Predict Result**](#11.Predict-Result)
# 
# 
# ---
# 
# # **1.Import h2o**
# 
# ---
# [**TOP**](#Outline-of-the-notebook)  
# Import the h2o Python module and H2OAutoML class and initialize a local H2O cluster.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import h2o
from h2o.automl import H2OAutoML
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
h2o.init()

# Any results you write to the current directory are saved as output.


# # **2.Load Data**
# 
# ---
# [**TOP**](#Outline-of-the-notebook)  
# 
# The goal here is to predict whether or not status.

# In[ ]:


# Load data into H2O
df = h2o.import_file("../input/train.csv")


# # **3.Statistics of Data**
# 
# ---
# [**TOP**](#Outline-of-the-notebook)  

# In[ ]:


df.describe()


# # **4.Split Train and target**
# 
# ---
# [**TOP**](#Outline-of-the-notebook)  

# In[ ]:


y = "C2"
x = df.columns
x.remove(y)
x.remove("C1")


# # **5.Automl Model Training**
# 
# ---
# [**TOP**](#Outline-of-the-notebook)  
# 
# Run AutoML, stopping after 10 models. The max_models argument specifies the number of individual (or "base") models, and does not include the two ensemble models that are trained at the end.

# In[ ]:


aml = H2OAutoML(max_runtime_secs = 990000, seed = 1,balance_classes=True)
aml.train(x = x, y = y, training_frame = df)


# # **6.Leader Board of Models**
# 
# ---
# [**TOP**](#Outline-of-the-notebook)  
# 
# Next, we will view the AutoML Leaderboard. Since we did not specify a leaderboard_frame in the H2OAutoML.train() method for scoring and ranking the models, the AutoML leaderboard uses cross-validation metrics to rank the models.
# 
# A default performance metric for each machine learning task (binary classification, multiclass classification, regression) is specified internally and the leaderboard will be sorted by that metric. In the case of binary classification, the default ranking metric is Area Under the ROC Curve (AUC). In the future, the user will be able to specify any of the H2O metrics so that different metrics can be used to generate rankings on the leaderboard.
# 
# The leader model is stored at aml.leader and the leaderboard is stored at aml.leaderboard.

# In[ ]:


lb = aml.leaderboard


# In[ ]:


lb.head()


# In[ ]:


lb.head(rows=lb.nrows)


# # **7.Ensemble Exploration**
# 
# ---
# [**TOP**](#Outline-of-the-notebook)  
# 
# ---
# 
# To understand how the ensemble works, let's take a peek inside the Stacked Ensemble "All Models" model. The "All Models" ensemble is an ensemble of all of the individual models in the AutoML run. This is often the top performing model on the leaderboard.

# In[ ]:


# Get model ids for all models in the AutoML Leaderboard
model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])
# Get the "All Models" Stacked Ensemble model
se = h2o.get_model([mid for mid in model_ids if "StackedEnsemble_AllModels" in mid][0])
# Get the Stacked Ensemble metalearner model
metalearner = h2o.get_model(se.metalearner()['name'])


# Examine the variable importance of the metalearner (combiner) algorithm in the ensemble. This shows us how much each base learner is contributing to the ensemble. The AutoML Stacked Ensembles use the default metalearner algorithm (GLM with non-negative weights), so the variable importance of the metalearner is actually the standardized coefficient magnitudes of the GLM.

# In[ ]:


metalearner.coef_norm()


# # **8.Model Importance**
# 
# ---
# [**TOP**](#Outline-of-the-notebook)  

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
metalearner.std_coef_plot()


# # **9.Save Leader Model**
# 
# ---
# [**TOP**](#Outline-of-the-notebook)  
# There are two ways to save the leader model -- binary format and MOJO format. If you're taking your leader model to production, then we'd suggest the MOJO format since it's optimized for production use.

# In[ ]:


aml.leader.model_id


# In[ ]:


h2o.save_model(aml.leader, path = "../working/")


# In[ ]:


model = h2o.load_model(aml.leader.model_id)


# # **10.Read test data**
# 
# ---
# [**TOP**](#Outline-of-the-notebook)  

# In[ ]:


df_test = h2o.import_file("../input/test.csv")
display(df_test.head())
df_test = df_test[1:,:]


# # **11.Predict Result**
# 
# ---
# [**TOP**](#Outline-of-the-notebook)  

# In[ ]:


predict = aml.predict(df_test)


# In[ ]:


predict.shape


# In[ ]:


submission = h2o.import_file("../input/sample_submission.csv")
submission['target1'] = predict
submission = submission.as_data_frame()
submission.columns = ['id', 'target1', 'target']
submission.pop('target1')
submission.to_csv("h2o.csv", index=False)
submission.head()

