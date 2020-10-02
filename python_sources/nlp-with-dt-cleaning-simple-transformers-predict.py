#!/usr/bin/env python
# coding: utf-8

# # [Real or Not? NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)

# ## This notebook is based on:
# * notebook [NLP with DT: Simple Transformers Research](https://www.kaggle.com/vbmokin/nlp-with-dt-simple-transformers-research) 
# * library [**simpletransformers**](https://github.com/ThilinaRajapakse/simpletransformers)
# * dataset [**NLP with Disaster Tweets - cleaning data**](https://www.kaggle.com/vbmokin/nlp-with-disaster-tweets-cleaning-data)

# In[ ]:


get_ipython().system('pip install --upgrade transformers')
get_ipython().system('pip install simpletransformers')


# In[ ]:


import numpy as np
import pandas as pd
import sklearn

import torch
from simpletransformers.classification import ClassificationModel

import warnings
warnings.simplefilter('ignore')


# In[ ]:


train_data = pd.read_csv('../input/nlp-with-disaster-tweets-cleaning-data/train_data_cleaning.csv')[['text', 'target']]
test_data = pd.read_csv('../input/nlp-with-disaster-tweets-cleaning-data/test_data_cleaning.csv')[['text']]


# In[ ]:


train_data


# In[ ]:


test_data


# In[ ]:


model_type = 'distilbert'
model_name = 'distilbert-base-uncased'
seed = 100
model_args =  {'fp16': False,
               'train_batch_size': 4,
               'gradient_accumulation_steps': 2,
               'do_lower_case': True,
               'learning_rate': 1e-05,
               'overwrite_output_dir': True,
               'manual_seed': seed,
               'num_train_epochs': 2}


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Training model and prediction\nmodel = ClassificationModel(model_type, model_name, args=model_args) \nmodel.train_model(train_data)\nresult, model_outputs, wrong_predictions = model.eval_model(train_data, acc=sklearn.metrics.accuracy_score)\ny_preds, _, = model.predict(test_data['text'])")


# In[ ]:


# Training model accuracy
print('accuracy =',result['acc'])


# In[ ]:


# Predicted data
y_preds[:20]


# In[ ]:


# Submission predicted data
test_data["target"] = y_preds
test_data.to_csv("submission.csv", index=False)

