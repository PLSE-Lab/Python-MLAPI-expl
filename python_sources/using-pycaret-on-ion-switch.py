#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import gc
import numpy as np
import pandas as pd


# # Files

# In[ ]:


train = pd.read_csv('../input/ion-switch-model-ready-data-frame-to-work-locally/train_ion_switch.csv')
test  = pd.read_csv('../input/ion-switch-model-ready-data-frame-to-work-locally/test_ion_switch.csv')


# # Random Sample

# In[ ]:


train_sample = train.sample(frac=0.005, random_state=25)


# In[ ]:


del train
gc.collect()


# # Installing PyCaret

# In[ ]:


get_ipython().system('pip install pycaret')


# # Import Classification

# In[ ]:


from pycaret.classification import *


# # Setup

# In[ ]:


clf1 = setup(data = train_sample, 
             target = 'open_channels',
             silent = True,
             remove_outliers = True,
             feature_selection = True)


# # LGBM

# In[ ]:


lgbm  = create_model('lightgbm')   


# # TUNING LGBM

# In[ ]:


tuned_lightgbm = tune_model('lightgbm')


# # EVALUATING LGBM

# In[ ]:


evaluate_model(tuned_lightgbm)


# # XGBOOST

# In[ ]:


xgb   = create_model('xgboost') 


# # TUNING XGBOOST

# In[ ]:


tuned_xgb = tune_model('xgboost')


# # EVALUATING XGBOOST

# In[ ]:


evaluate_model(tuned_xgb)


# # PREDICTIONS

# In[ ]:


pred_lgbm = predict_model(tuned_lightgbm, data=test)
pred_lgbm['open_channels'] = pred_lgbm['Label']

pred_xgb = predict_model(tuned_xgb, data=test)
pred_xgb['open_channels'] = pred_xgb['Label']


# # A SIMPLE BLEND

# In[ ]:


blend = blend_models(estimator_list = [tuned_lightgbm,tuned_xgb])
pred_blend = predict_model(blend, data=test)
#pred_blend.to_csv('pred_blend.csv',index=False)
pred_blend.head()


# # LGBM SUBMISSION

# In[ ]:


sub = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv")

submission = pd.DataFrame()
submission['time']  = sub['time']
submission['open_channels'] = pred_lgbm['open_channels']
submission['open_channels'] = submission['open_channels'].round(decimals=0)
submission['open_channels'] = submission['open_channels'].astype(int)
submission.to_csv('submission_lgbm.csv', float_format='%0.4f', index = False)


# # XGB SUBMISSION

# In[ ]:


sub = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv")

submission = pd.DataFrame()
submission['time']  = sub['time']
submission['open_channels'] = pred_xgb['open_channels']
submission['open_channels'] = submission['open_channels'].round(decimals=0)
submission['open_channels'] = submission['open_channels'].astype(int)
submission.to_csv('submission_xgb.csv', float_format='%0.4f', index = False)


# # Conclusion
# 
#  - PyCaret is convenient to see general image but it takes too much time when using compare and evaluation.
#  - I had to use a very small fraction just to see it.
