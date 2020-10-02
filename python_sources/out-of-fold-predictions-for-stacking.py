#!/usr/bin/env python
# coding: utf-8

# You can use training set ouf of fold predictions and out of fold predictions on test set.
# 
# All the models are described in this repo https://github.com/neptune-ml/kaggle-toxic-starter .
# 
# Some usual suspects like gru, and lstm on different embeddings get 0.985+

# In[4]:


get_ipython().system('ls ../input/out-of-fold-predictions/single_model_predictions_03092018/single_model_predictions_03092018/')


# For example the predictions on train for the GRU model trained on fasttext look like this:

# In[5]:


import pandas as pd

train_oof_predictions = pd.read_csv('../input/out-of-fold-predictions/single_model_predictions_03092018/single_model_predictions_03092018/fasttext_gru_predictions_train_oof.csv')
train_oof_predictions.head()


# ![](http://)use 'fold_id' column to do your train/test split for the second level models

# In[6]:


test_oof_predictions = pd.read_csv('../input/out-of-fold-predictions/single_model_predictions_03092018/single_model_predictions_03092018/fasttext_gru_predictions_test_oof.csv')
test_oof_predictions.head()


# Again 'fold_id' column can/should be used in setting up the validation regime

# In[ ]:




