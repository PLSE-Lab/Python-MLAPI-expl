#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import os

from scipy.stats import rankdata

LABELS = ["target"]
print(os.listdir('../input/'))


# In[7]:


# inspired by [...]
predict_list = []

predict_list.append(pd.read_csv('../input/90-lines-solution-0-901-fast/lgb_submission.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/eda-pca-lgbm-santander-transactions/submission34.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/eda-pca-simple-lgbm-on-kfold-technique/submission26.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/lgb-2-leaves-augment/lgb_submission.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/lightgbm-with-data-augmentation/2019-04-01_07_56_sub.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/santander-augment-to-the-rescue/submission.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/others/output_v3.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/others/output_v2.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/others/output_v1.csv')[LABELS].values)


# In[ ]:


print("Rank averaging on ", len(predict_list), " files")
predictions = np.zeros_like(predict_list[0])
for predict in predict_list:
    for i in range(1):
        predictions[:, i] = np.add(predictions[:, i], rankdata(predict[:, i])/predictions.shape[0])  
predictions /= len(predict_list)

submission = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')
submission[LABELS] = predictions
submission.to_csv('super_blend.csv', index=False)


# # Inspired by:
# 
# [0] https://www.kaggle.com/jesucristo/simple-blend-my-best-score
# 
# [1] https://www.kaggle.com/jesucristo/90-lines-solution-0-901-fast
# 
# [2] https://www.kaggle.com/sagarprasad/customer-transaction-prediction
# 
# [3] https://www.kaggle.com/roydatascience/eda-pca-lgbm-santander-transactions
# 
# [4] https://www.kaggle.com/roydatascience/eda-pca-simple-lgbm-on-kfold-technique
# 
# [5] https://www.kaggle.com/jiweiliu/lgb-2-leaves-augment
# 
# [6] https://www.kaggle.com/fdewes/lgbm-training-augmentation
# 
# [7] https://www.kaggle.com/omgrodas/lightgbm-with-data-augmentation
# 
# [8] https://www.kaggle.com/subhamsharma96/santander-augment-to-the-rescue
