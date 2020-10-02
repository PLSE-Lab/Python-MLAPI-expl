#!/usr/bin/env python
# coding: utf-8

# # Demonstration Kernel with fast-bert
# 
# This notebook is an annotated copy of https://github.com/kaushaltrivedi/fast-bert/blob/master/sample_notebooks/toxic-multilabel-lib.ipynb
# 
# 

# # Install Packages
# fast-bert (will install bert)
# 
# apex (must be compiled from source) **(can take ~5mins)**

# In[ ]:


get_ipython().system('pip install fast-bert')


# In[ ]:


get_ipython().system('git clone https://github.com/NVIDIA/apex')


# In[ ]:


get_ipython().run_line_magic('cd', 'apex')
get_ipython().system('pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .')
get_ipython().run_line_magic('cd', '..')


# # Data Handling
# fast-bert can deal directly with text, pre-processing will be handled. CSV file needs to be adapted, as it expects the multi-class labels to be given as dummy variables, while in original format we have a single column with the label

# In[ ]:


import pandas as pd
import os


# In[ ]:


get_ipython().system('mkdir data')
DATA_FOLDER = 'data'


# In[ ]:


train_df = pd.read_csv('../input/train.csv', index_col='id')
val_df = pd.read_csv('../input/valid.csv', index_col='id')
test_df = pd.read_csv('../input/test.csv', index_col='id') 


# ## Labels
# fast-bert needs the list of labels in a flat file

# In[ ]:


label_cols = list(pd.get_dummies(train_df['label']).columns)
with open(os.path.join(DATA_FOLDER, 'labels.csv'), 'w') as f:
    f.write('\n'.join(label_cols))


# In[ ]:


def to_fastbert(df:pd.DataFrame, name:str):
  d = pd.get_dummies(df['label'])
  d['text'] = df['text']
  d.to_csv(os.path.join(DATA_FOLDER, f'{name}.csv'), index=True, index_label=['id'])
  


# In[ ]:


for x,n in zip([train_df, val_df], ['train', 'val']):
  to_fastbert(x, n)


# # Get a Pre-trained BERT

# In[ ]:


#!wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip


# In[ ]:


#!unzip multi_cased_L-12_H-768_A-12.zip


# ## Convert BERT from Tensorflow to PyTorch (if using a local version)

# In[ ]:


#!git clone https://github.com/huggingface/pytorch-pretrained-BERT.git


# In[ ]:


#!python pytorch-pretrained-BERT/pytorch_pretrained_bert/convert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path multi_cased_L-12_H-768_A-12/bert_model.ckpt --bert_config_file multi_cased_L-12_H-768_A-12/bert_config.json --pytorch_dump_path multi_cased_L-12_H-768_A-12/pytorch_model.bin


# # Code for the Classifier based on BERT

# ## Imports and setup

# In[ ]:


from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTraining, BertConfig, BertForMaskedLM, BertForSequenceClassification
from pathlib import Path
import torch

from fastai.text import Tokenizer, Vocab
import pandas as pd
import collections
import os
from tqdm import tqdm, trange
import sys
import random
import numpy as np
import apex
from sklearn.model_selection import train_test_split

import datetime
    
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.optimization import BertAdam

from fast_bert.modeling import BertForMultiLabelSequenceClassification
from fast_bert.data import BertDataBunch, InputExample, InputFeatures, MultiLabelTextProcessor, convert_examples_to_features
from fast_bert.learner import BertLearner
from fast_bert.metrics import accuracy_multilabel, accuracy_thresh, fbeta, roc_auc


# In[ ]:


torch.cuda.empty_cache()
pd.set_option('display.max_colwidth', -1)
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')


# ## Set up path for data

# In[ ]:


DATA_PATH = Path('data/')
LABEL_PATH = Path('data/')

MODEL_PATH=Path('models/')
LOG_PATH=Path('logs/')
MODEL_PATH.mkdir(exist_ok=True)

model_state_dict = None

FINETUNED_PATH = None
# model_state_dict = torch.load(FINETUNED_PATH)

LOG_PATH.mkdir(exist_ok=True)


# For the pre-trained BERT model, 2 choices:
# * Provide the path to a saved / downloaded model (like we did above, in folder `multi_cased_L-12_H-768_A-12`)
# * Give the name of a standard model, in the list `[bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese]`
# 
# These two options are valid when using `BertTokenizer.from_pretrained()` and `BertLearner.from_pretrained_model()` factory methods

# In[ ]:


# If using a local version, download and convert by uncommenting cells 10,11,12,13
#BERT_PRETRAINED_PATH = Path('multi_cased_L-12_H-768_A-12/')

# We'll let the library get a plus-n-play model 
BERT_PRETRAINED_PATH = 'bert-base-multilingual-cased'


# ## All arguments for training the model

# In[ ]:


args = {
    "run_text": "amazon pet review",
    "train_size": -1,
    "val_size": -1,
    "log_path": LOG_PATH,
    "full_data_dir": DATA_PATH,
    "data_dir": DATA_PATH,
    "task_name": "Amazon Pet Review",
    "no_cuda": False,
    "bert_model": BERT_PRETRAINED_PATH,
    "output_dir": MODEL_PATH/'output',
    "max_seq_length": 512,
    "do_train": True,
    "do_eval": True,
    "do_lower_case": False,
    "train_batch_size": 16,
    "eval_batch_size": 16,
    "learning_rate": 5e-6,
    "num_train_epochs": 4.0,
    "warmup_proportion": 0.1,
    "no_cuda": False,
    "local_rank": -1,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "optimize_on_cpu": False,
    "fp16": True,
    "loss_scale": 128
}


# ## Logging

# In[ ]:


import logging

logfile = str(LOG_PATH/'log-{}-{}.txt'.format(run_start_time, args["run_text"]))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[
        logging.FileHandler(logfile),
        logging.StreamHandler(sys.stdout)
    ])

logger = logging.getLogger()


# In[ ]:


logger.info(args)


# ## Instantiate a Tokenizer from a pre-trained BERT
# **Careful with lower_case**

# In[ ]:


tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_PATH, do_lower_case=args['do_lower_case'])


# In[ ]:


device = torch.device('cuda')
if torch.cuda.device_count() > 1:
    multi_gpu = True
else:
    multi_gpu = False


# ## Load the data from prepared dataset
# 
# The `BertDataBunch` organizes the delivery of data to the model during training.
# 
# See https://github.com/kaushaltrivedi/fast-bert/blob/master/fast_bert/data.py
# 
# Argument per argument:
# 
# - `data_dir`: folder where to look for data files
# - `label_dir`: folder with the labels file
# - `tokenizer`: the tokenizer that will be used to pre-process the text before submitting to BERT embedding layer
# - `train_file`, `val_file`. `test_data`: csv files with the data. in multi-class situation, the labels must be 1-hot encoded with dummy columns
# - `label_file`: the file with the list of labels. Omitted in our case, the default is `labels.csv`
# - `text_col`: in the dataframes made from the data csv files, what is the name of the column with the text to be classified
# - `label_col`: in the dataframes made from the data csv files, what are the names of the columns with the 1-hot encoded classification labels
# - `bs`: batch size for training
# - `max_len`: maximum sequence length for BERT input
# - `multi_gpu`: True if training will happen over multiple gpu
# - `multi_label`: True is the classification task is multi-label, False if binary classification
# 
# The resulting object will be used when instantiating the BertLearner that will actually do the training.
# 
# **Can take some time, as it is pre-processing all text, using the provided Tokenizer**

# In[ ]:


databunch = BertDataBunch(args['data_dir'], LABEL_PATH, tokenizer, train_file='train.csv', val_file='val.csv',
                          test_data=list(test_df['text'].values),
                          text_col="text", label_col=label_cols,
                          bs=args['train_batch_size'], maxlen=args['max_seq_length'], 
                          multi_gpu=multi_gpu, multi_label=True)
databunch.save()


# In[ ]:


num_labels = len(databunch.labels)


# ## Metrics to report at evaluation time

# In[ ]:


from functools import partial

metrics = []
metrics.append({'name': 'accuracy_thresh', 'function': accuracy_thresh})
metrics.append({'name': 'roc_auc', 'function': roc_auc})
metrics.append({'name': 'F1', 'function': partial(fbeta, beta=1)})
metrics.append({'name': 'accuracy_single', 'function': accuracy_multilabel})


# ## Create an estimator and train it
# This is now a complete model, inluding a BERT model for feature-generation, on top of which a classifier is added.
# It is 1 fully connected layer, that takes the embedding for the `<CLS>` token as input, and outputs as many logits as labels.
# 
# There is no way in current code to force another kind of classifier (more layers, etc...). The way to do that would be to inheritate the `BertForMultiLabelSequenceClassification` class and add parameters that describe the Classifier Neural Network, or to suggest a PR to the author with a modified Base class.
# See https://github.com/kaushaltrivedi/fast-bert/blob/master/fast_bert/modeling.py 
# 
# We use all the created objects...
# - the databunch, to feed data into the model
# - the path to the BERT pre-trained, for weight initialization
# - the metrics, training device, logger
# - the path where the fine-tuned model will be saved
# 
# Once the `BertLearner` is created, the `fit()` method is called for the training.

# In[ ]:


learner = BertLearner.from_pretrained_model(databunch, BERT_PRETRAINED_PATH, metrics, device, logger, 
                                            finetuned_wgts_path=FINETUNED_PATH, 
                                            is_fp16=args['fp16'], loss_scale=args['loss_scale'], 
                                            multi_gpu=multi_gpu,  multi_label=True)


# **(45mins per epoch, with GPU activated)**

# In[ ]:


learner.fit(4, lr=args['learning_rate'], schedule_type="warmup_cosine_hard_restarts")


# # Predict the TEST set

# In[ ]:


preds = learner.predict_batch()


# In[ ]:


test_df['label'] = [max(x, key=lambda z: z[1])[0] for x in preds]
test_df['label'].to_csv('fast_bert_submission.csv', index=True, index_label=['id'], header=True)


# # Clean

# In[ ]:


get_ipython().system('rm -Rf apex')

