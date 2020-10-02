#!/usr/bin/env python
# coding: utf-8

# Inspired the super blend technique by Saurabh Kumar (https://www.kaggle.com/saurabh502/why-no-blend),
# 
# I added my kernel output to the existing set. I had used Stratified KFolds techniques with Light GBM Classifier. 
# 
# https://www.kaggle.com/roydatascience/light-gbm-on-stratified-k-folds-malwares
# 
# **NEW UPDATE**
# 
# I have also  added outputs from Hung The Nguyen Kernel (https://www.kaggle.com/hung96ad/lightgbm) 

# In[ ]:


import numpy as np
import pandas as pd
import os

from scipy.stats import rankdata

LABELS = ["HasDetections"]


# In[ ]:


predict_list = []


predict_list.append(pd.read_csv("../input/hung-the-nguyen/submission_lgbm_5.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/hung-the-nguyen/submission_lgbm_6.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/hung-the-nguyen/submission_lgbm_7.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/hung-the-nguyen/submission_lgbm_8.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/hung-the-nguyen/submission_lgbm_9.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/new-blend/super_blend.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/new-blend/super_blend.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/new-blend/super_blend.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/new-blend/super_blend.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/new-blend/super_blend.csv")[LABELS].values)


# In[ ]:


print("Rank averaging on ", len(predict_list), " files")
predictions = np.zeros_like(predict_list[0])
for predict in predict_list:
    for i in range(1):
        predictions[:, i] = np.add(predictions[:, i], rankdata(predict[:, i])/predictions.shape[0])  
predictions /= len(predict_list)

submission = pd.read_csv('../input/microsoft-malware-prediction/sample_submission.csv')
submission[LABELS] = predictions
submission.to_csv('super_blend.csv', index=False)


# In[ ]:


submission.head()

