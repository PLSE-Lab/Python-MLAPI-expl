#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# https://www.kaggle.com/c/google-quest-challenge/discussion/123770


# In[ ]:


get_ipython().system('pip install ../input/sacremoses/sacremoses-master/ > /dev/null')


# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import argparse
import collections
import datetime
import gc
import glob
import logging
import math
import operator
import os 
import pickle
import pkg_resources
import random
import re
import scipy.stats as stats
import seaborn as sns
import shutil
import sys
import time
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from contextlib import contextmanager
from collections import OrderedDict
# from nltk.stem import PorterStemmer
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import KFold, GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.nn import CrossEntropyLoss, MSELoss
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import (Dataset,DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
# from tqdm import tqdm, tqdm_notebook, trange
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
tqdm.pandas()
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings
warnings.filterwarnings('ignore')
# from apex import amp


# In[ ]:


sys.path.insert(0, "../input/transformers/transformers-master/")

from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from transformers import (WEIGHTS_NAME, BertConfig, BertModel, PreTrainedModel, BertPreTrainedModel,
                          AlbertModel, AlbertForSequenceClassification, BertForSequenceClassification,
                          AlbertTokenizer, AlbertConfig,
                                  BertForQuestionAnswering, BertTokenizer,
                                  XLMConfig, XLMForQuestionAnswering,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForQuestionAnswering,
                                  XLNetTokenizer, DistilBertTokenizer, DistilBertModel)

from transformers import AdamW # , WarmupLinearSchedule
from transformers.tokenization_bert import (BasicTokenizer,
                                                    whitespace_tokenize)

from transformers.modeling_bert import BertLayer, BertEmbeddings, BertEncoder, BertPooler

SEED = 1129

def seed_everything(seed=1129):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(SEED)


# In[ ]:


class BertForSequenceClassification_v2(BertPreTrainedModel):

    def __init__(self, config, num_labels=30):

        super(BertForSequenceClassification_v2, self).__init__(config)

        # config.output_hidden_states=True (make sure)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None, 
                extra_feats=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        # sequence_output = outputs[0]
        # pooled_output = outputs[1]

        hidden_states = outputs[2] #hidden_states: 12 layers tuples each is of (batch_size, sequence_length, hidden_size) + embedding``
        # print(seq[-1].shape, seq[-1][:, 0].shape)

        # we are taking zero because in my understanding that's the [CLS] token...
        # idea is to pool last 4 layers as well instead of just the last one, since it's too close to the output
        # layers, it might not be that efficient as it's more regulated by the o/p's..

        h12 = hidden_states[-1][:, 0].reshape((-1, 1, 768))
        h11 = hidden_states[-2][:, 0].reshape((-1, 1, 768))
        h10 = hidden_states[-3][:, 0].reshape((-1, 1, 768))
        h9  = hidden_states[-4][:, 0].reshape((-1, 1, 768))

        all_h = torch.cat([h9, h10, h11, h12], 1)
        mean_pool = torch.mean(all_h, 1)

        pooled_output = self.dropout(mean_pool)
        logits = self.classifier(pooled_output)
        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return logits


# In[ ]:


# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, additional_special_tokens = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[ANS]", "[QBODY]"])

# Tokenize input (dummy example)
text = "[CLS] Who was Jim Henson ? [QBODY] [ANS] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)
#outputs
print(tokenized_text)
#['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[ANS]', '[QBODY]', 'jim', 'henson', 'was', 'a', 'puppet', '##eer', '[SEP]']


# In[ ]:


# tokenizer.vocab["[ANS]"] = -1
# tokenizer.vocab["[QBODY]"] = -1


# In[ ]:





# In[ ]:


# Just in case someone is adding new tokens to transformers (Hugging Face)
### Let's load a model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
### Do some stuff to our model and tokenizer
# Ex: add new tokens to the vocabulary and embeddings of our model (another way)
tokenizer.add_tokens(['[ANS]', '[QBODY]'])
model.resize_token_embeddings(len(tokenizer))


# In[ ]:


tokenizer.vocab["who"]


# In[ ]:


train(model)


# In[ ]:


# Train our model
# train(model)

### Now let's save our model and tokenizer to a directory
model.save_pretrained('/kaggle/working')
tokenizer.save_pretrained('/kaggle/working')
### Reload the model and the tokenizer
model = BertForSequenceClassification.from_pretrained('/kaggle/working')
tokenizer = BertTokenizer.from_pretrained('/kaggle/working')


# In[ ]:


text = "[CLS] Who was Jim Henson ? [QBODY] [ANS] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)
#outputs
print(tokenized_text)


# In[ ]:


for i in tokenized_text:
    try:
        print(tokenizer.vocab[i])
    except KeyError:
        print(i)


# In[ ]:




