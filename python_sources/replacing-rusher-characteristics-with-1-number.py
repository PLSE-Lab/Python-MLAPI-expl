#!/usr/bin/env python
# coding: utf-8

# # Replacing rusher characteristics with 1 number
# 
# Here we aimed to create a submodel only on a few rusher player characteristics to examine if they indeed can have a significant information gain on the problem outcome. But more importantly, we want to look at the variance of the yards prediction of this submodel compred to that of the whole dataset.
# 
# We use CV10 in order to get predictions for each entry in the training set, rather than splitting on a train and test.

# In[ ]:


import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy import integrate
import itertools 
import pylab as plt
import datetime
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px
import math
import re
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.grid.grid_search import H2OGridSearch


# ### Player physical traits model - replace PlayerHeight, PlayerWeight, PlayerBirthDate, PlayerCollegeName by 1 continuous variable

# In[ ]:


dataset = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv", low_memory=False)


# In[ ]:


def feet_inch_to_cm(height_fi):
    foot = 30.48
    inch = 2.54
    return int(height_fi.split('-')[0]) * foot + int(height_fi.split('-')[1]) * inch


# In[ ]:


dataset = dataset[dataset['NflId']==dataset['NflIdRusher']][['GameId', 'PlayId', 'PlayerHeight', 'PlayerWeight', 'PlayerBirthDate', 'PlayerCollegeName', 'Yards']]
dataset['Age']=2019 - pd.DatetimeIndex(dataset['PlayerBirthDate']).year
dataset['PlayerHeight_cm']= dataset['PlayerHeight'].apply(feet_inch_to_cm)
dataset['PlayerWeight_kg']= dataset['PlayerWeight']*0.453592


# In[ ]:


dataset=dataset[['GameId', 'PlayId', 'PlayerCollegeName', 'Yards', 'Age', 'PlayerHeight_cm', 'PlayerWeight_kg']].reset_index(drop=True)


# In[ ]:


h2o.init()
train = h2o.H2OFrame(dataset)


# In[ ]:


# # GBM hyperparameters
# gbm_params1 = {'ntrees': [300,500,700],
#                 'max_depth': [1, 2, 3,4],
#                 'min_rows': [2,5,10]}

# # Train and validate a cartesian grid of GBMs
# gbm_grid1 = H2OGridSearch(model=H2ORandomForestEstimator,
#                           grid_id='gbm_grid1',
#                           hyper_params=gbm_params1)
# gbm_grid1.train(x=['PlayerCollegeName', 'Age', 'PlayerHeight_cm', 'PlayerWeight_kg'], y='Yards',
#                 training_frame=train,
#                 seed=1)
# gbm_gridperf1 = gbm_grid1.get_grid(sort_by='mae')
# gbm_gridperf1


# In[ ]:


model = H2ORandomForestEstimator(ntrees=300, max_depth=3, min_rows=2, nfolds=10, seed = 1, keep_cross_validation_predictions=True)

# Train model
model.train(x=['PlayerCollegeName', 'Age', 'PlayerHeight_cm', 'PlayerWeight_kg'], y='Yards', training_frame=train)
cv_predictions = model.cross_validation_holdout_predictions()
train = train.as_data_frame().join(cv_predictions.as_data_frame())


# In[ ]:


model.varimp(True)


# In[ ]:


train.head()


# From the histogram below, it looks like rusher characteristics do have some information gain, but the variance is not anywhere near that of the dataset.
# We used the output of this model as a feature in a final model, which did result in an almost insignificant improvement of CRPS.

# In[ ]:


plt.figure(figsize=[20,10])
train['Yards'].hist(bins=90, label='Test Set')
train['predict'].hist(bins=90, label='Predictions')
plt.legend(loc='upper right')

