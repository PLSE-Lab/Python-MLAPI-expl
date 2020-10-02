#!/usr/bin/env python
# coding: utf-8

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


# ### Listing down useful single models

# ### LGB / XGB 
# * m1 = k-fold xgb [0.9383] : https://www.kaggle.com/artkulak/ieee-fraud-simple-baseline-0-9383-lb
# * m2 = h20 automl [0.9395] : https://www.kaggle.com/tunguz/ieee-with-h2o-automl
# * m3 = lgb + fe [0.9408] : https://www.kaggle.com/kyakovlev/ieee-ground-baseline/output
# * m4 = lgb + bayesian tuning [ ] : [https://www.kaggle.com/vincentlugat/ieee-lgb-bayesian-opt/output]
# * m5 = lgb + feature selection [.9419]: [https://www.kaggle.com/nroman/lgb-single-model-lb-0-9419/output]
# * m6 = lgb + fe [0.9429] : [https://www.kaggle.com/timon88/lgbm-baseline-small-fe-no-blend/notebook]
# * m7 = lgb + fe : [0.9415] https://www.kaggle.com/plasticgrammer/ieee-cis-fraud-detection-playground/output
# * m8 = xgb + fe [] : https://www.kaggle.com/iasnobmatsu/xgb-model-with-feature-engineering
# * m9 = lgbm + gpu + bayesian hyperparameter opt [0.9430] : https://www.kaggle.com/nicapotato/gpyopt-hyperparameter-optimisation-gpu-lgbm/output
# * m10 = lgb in R [0.9437] : https://www.kaggle.com/andrew60909/lgb-starter-r/output
# * m11 = lgb + fe [0.9441] : https://www.kaggle.com/kyakovlev/ieee-ground-baseline-make-amount-useful-again
# * m12 = xgb + fe [0.9442] : https://www.kaggle.com/krishonaveen/xtreme-boost-and-feature-engineering/output
# * m13 = lgb + fe [0.9449] : https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm-corrected/output
# * m14 = hst lgb + fe in R [0.94551] : https://www.kaggle.com/duykhanh99/hust-lgb-feature-engineering-with-r
# * m15 = lgbm + fe in R [0.9452] : https://www.kaggle.com/duykhanh99/lightgbm-fe-with-r
# * m17 = lgb [0.9463]: https://www.kaggle.com/roydatascience/light-gbm-with-complete-eda/output
# * m18 = lgb + new features [0.9467] : https://www.kaggle.com/gunesevitan/lightgbm-some-new-features
# * m19 = lgb + fe in R [0.9469] : https://www.kaggle.com/abednadir/best-r-score
# * m20 = lgb + group kfold cv [0.9483] : https://www.kaggle.com/kyakovlev/ieee-lgbm-with-groupkfold-cv
# 
# ### Catboost
# * m16 = catboost [0.9454] : https://www.kaggle.com/pipboyguy/catboost-and-eda/output
# 
# 
# ### NN
# * m0 = nn + focal loss expt [0.92] : https://www.kaggle.com/abazdyrev/keras-nn-focal-loss-experiments

# In[ ]:


### LGB
m15 = pd.read_csv('/kaggle/input/ieee-top-models-blend/m15.csv')
m17 = pd.read_csv('/kaggle/input/ieee-top-models-blend/m17.csv')
m18 = pd.read_csv('/kaggle/input/ieee-top-models-blend/m18.csv')
m19 = pd.read_csv('/kaggle/input/ieee-top-models-blend/m19.csv')
m20 = pd.read_csv('/kaggle/input/ieee-top-models-blend/m20.csv')
### Catboost
m16 = pd.read_csv('/kaggle/input/ieee-top-models-blend/m16.csv')
### NN
m0 = pd.read_csv('/kaggle/input/ieee-top-models-blend/m0.csv')

submission = pd.read_csv("/kaggle/input/ieee-fraud-detection/sample_submission.csv")
submission.head()


# In[ ]:


m18.head()


# In[ ]:


submission['isFraud'] = (0.20*m15.isFraud) +                         (0.20*m17.isFraud) +                         (0.20*m18.isFraud) +                         (0.20*m19.isFraud) +                         (0.10)*m20.isFraud +                         (0.10*m16.isFraud) +                         (0.0*m0.isFraud)       
                        
submission.to_csv('my_blend_5.csv', index=False)

