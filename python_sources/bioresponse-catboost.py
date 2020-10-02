#!/usr/bin/env python
# coding: utf-8

# ## Predicting a biological response of molecules from their chemical properties using CatBoost 
# 
# 
# 
# ***************
# 
# 
# The goal of this project is to relate molecular information to an actual biological response based on the dataset.
# 
# Each row in this dataset represents a molecule. The first column contains experimental data describing an actual biological response; the molecule was seen to elicit this response (1), or not (0). 
# 
# The remaining columns represent molecular descriptors (d1 through d1776), these are calculated properties that can capture some of the characteristics of the molecule - for example size, shape, or elemental constitution. 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# ## Import libraries

# In[ ]:


import math
import scipy
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import matplotlib.image as mpimg    # to check images
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Upload data

# In[ ]:


train_data = pd.read_csv('../input/bioresponse/train.csv', header=0, sep=',')
test_data = pd.read_csv('../input/bioresponse/test.csv')
X_train = train_data.iloc[:,1:]
Y_train = train_data.Activity
X_test = test_data.iloc[:,:]


# 
# ********************
# 

# # CatBoost

# 
# 
# * CatBoost is a machine learning algorithm that uses gradient boosting on decision trees. 
# 
# * It is available as an open source library.
# 
# 
# ******************************
# 
# 
# ![](https://avatars.mds.yandex.net/get-bunker/56833/dba868860690e7fe8b68223bb3b749ed8a36fbce/orig)
# 
# 
# *****************************
# 
# 
# 
# ### See some videos about CatBoost:
# 
# 
# 
# <div class="alert alert-block alert-info">
# <img src='https://i.imgur.com/H6AnLaj.png' width='90' align='left'></img>
# <p><a href='https://www.youtube.com/watch?v=s8Q_orF4tcI'>Catboost: Open-source Gradient Boosting Library!</a></p>
# <p>Yandex Research</p>
# </div>
# 
# 
# ******************************
# 
# 
# 
# 
# 
# <div class="alert alert-block alert-info">
# <img src='https://i.imgur.com/H6AnLaj.png' width='90' align='left'></img>
# <p><a href='https://www.youtube.com/watch?v=dvZLk7LxGzc'>CatBoost VS XGboost - It's Modeling Cat Fight Time! Welcome to 5 Minutes for Data Science!</a></p>
# <p>Manuel Amunategui</p>
# </div>
# 
# 
# ******************************
# 
# 
# 
# ### References: 
# 
# 
# https://catboost.ai/
# 
# https://github.com/catboost/catboost
# 
# 
# 
# Anna Veronika Dorogush, Andrey Gulin, Gleb Gusev, Nikita Kazeev, Liudmila Ostroumova Prokhorenkova, Aleksandr Vorobev 
# "Fighting biases with dynamic boosting". arXiv:1706.09516, 2017.
# https://arxiv.org/abs/1706.09516
# 
# 
# Anna Veronika Dorogush, Vasily Ershov, Andrey Gulin 
# "CatBoost: gradient boosting with categorical features support". Workshop on ML Systems at NIPS 2017.
# http://learningsys.org/nips17/assets/papers/paper_11.pdf
# 
# 

# ## CatBoostClassifier
# 

# In[ ]:


import catboost
from catboost import CatBoostClassifier
modelC550 = CatBoostClassifier(verbose=0, n_estimators=550000)
modelC550.fit(X_train, Y_train)
predicted_probsC550 = modelC550.predict_proba(X_test)


# ## Making predictions
# 

# In[ ]:


predicted_probs = predicted_probsC550


# In[ ]:


PredictedProbability = predicted_probs[:,1]
MoleculeId = np.array(range(1,len(PredictedProbability)+1))
result=pd.DataFrame()
result['MoleculeId'] = MoleculeId
result['PredictedProbability'] = PredictedProbability
result.to_csv('!Averaging_Ensemble_Bioresponse.csv',index=None)


# 
# 
# ***********************
# 
# 

# 
# 
# ## Notebooks
# 
# https://www.kaggle.com/sgladysh/bioresponse-ensemble-weighted-averaging
# 
# https://www.kaggle.com/sgladysh/bioresponse-ensemble-averaging
# 
# https://www.kaggle.com/sgladysh/bioresponse-catboost
# 
# https://www.kaggle.com/sgladysh/xgboost-bioresponse
# 
# https://www.kaggle.com/sgladysh/sklearn-gb-bioresponse
# 
# https://www.kaggle.com/sgladysh/lightgbm-bioresponse
# 
# 
# 
# ********************

# 
