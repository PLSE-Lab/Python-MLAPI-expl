#!/usr/bin/env python
# coding: utf-8

# # Exploring unstable features that don't work as well if test is time split (vs regular cv)
# 
# 
# ## For more info on splits for this competition see these kernels: 
# 
# - [2 months train, 1 month public, 1 day private?](https://www.kaggle.com/rquintino/2-months-train-1-month-public-1-day-private) 
# - [Time Split Validation - Malware - [0.68] kernel](https://www.kaggle.com/cdeotte/time-split-validation-malware-0-68)
# 

# In[ ]:


import numpy as np
import pandas as pd
import os

pd.options.display.max_rows = 100


# ## Load both pre calculated CV AUC and Time Split AUC from single feature models
# 
# - Time split considered was < 2018-09-20 for train and  >=2018-09-20 for test.

# In[ ]:


df_cv_score=pd.read_csv("../input/classification-auc-per-single-feature/cv_feature_results.csv").groupby("feature")["cv_score"].mean().reset_index().sort_values("cv_score",ascending=False)
df_timesplit_score=pd.read_csv("../input/classification-auc-per-single-feature-time-split/time_split_feature_results.csv")
df_scores=pd.merge(df_cv_score,df_timesplit_score).drop(columns="index")
df_scores["time_split_vs_cv"]=df_scores.time_split_score-df_scores.cv_score
df_scores.sort_values("time_split_vs_cv",ascending=True)


# In[ ]:




