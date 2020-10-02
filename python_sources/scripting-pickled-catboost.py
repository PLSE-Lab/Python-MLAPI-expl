#!/usr/bin/env python
# coding: utf-8

# # Hello
# 
#  This is just a simple kernel so I can demonstrate utility scripting. For the [utility scripting competition](https://www.kaggle.com/general/109651). This runs a hyperopt search on Catboost, creates predictions, outputs feature importances, etc...
#  
#  Used scripts:
#  
#  - [Optimizer utils](https://www.kaggle.com/donkeys/optimizer-utils)
#  - [Optimizers](https://www.kaggle.com/donkeys/optimizers) built on those utils
# 
# Needs some work but feedback and improvements are welcome.. :)

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


data_path = "/kaggle/input/simple-preprocessing/"


# In[ ]:


import pickle

with open(data_path+"/df_train.pkl", "rb") as myfile:
    df_train = pickle.load(myfile)


# In[ ]:


with open(data_path+"/df_test.pkl", "rb") as myfile:
    df_test = pickle.load(myfile)


# How many optimization iterations to run:

# In[ ]:


#How many iterations of Hyperopt search to run
#if you enable GPU below with use_gpu=True (and in kernel), this can be much higher. 
# But Kaggle limits GPU use so much, I use CPU here :) (or maybe just a quickie..)
N_ROUNDS_CATB = 25
#This is just the N folds for the final train/predict after running the hyperopt search
N_FOLDS = 10


# In[ ]:


df_train.columns


# In[ ]:


df_test.columns


# In[ ]:


target = df_train["target"]


# In[ ]:


df_train.drop("target", inplace=True, axis=1)


# In[ ]:


from optimizers import *

catb_opt = CatboostOptimizer()
catb_opt.n_trials = N_ROUNDS_CATB
catb_opt.n_folds = N_FOLDS
catb_opt.use_gpu = True
catb_opt.cat_features = categorical_indices(df_train)
X_cols = df_train.columns
catb_results = catb_opt.classify_binary(X_cols, df_train, df_test, target)


# In[ ]:


submission = pd.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv")
submission.head()


# In[ ]:


submission["target"] = catb_results.predictions[:, 1]
submission.head()


# In[ ]:


submission.to_csv("submission_catb.csv", index=False)


# In[ ]:


get_ipython().system('head submission_catb.csv')

