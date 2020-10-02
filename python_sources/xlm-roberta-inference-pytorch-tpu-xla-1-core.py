#!/usr/bin/env python
# coding: utf-8

# # XLM-RoBERTa inference
# 
# ## Please upvote if you found this helpful
# 
# This is just some quick an dirty inference for [this kernel](https://www.kaggle.com/tanlikesmath/xlm-roberta-pytorch-xla-tpu/). Someone will probably come out with a 8-core inference kernel.

# In[ ]:


get_ipython().system('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
get_ipython().system('python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev')


# In[ ]:


import os
import torch
import pandas as pd
from scipy import stats
import numpy as np

from tqdm import tqdm
from collections import OrderedDict, namedtuple
import torch.nn as nn
from torch.optim import lr_scheduler
import joblib

import logging
import transformers
import sys


# In[ ]:


import warnings
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
import warnings
import time

warnings.filterwarnings("ignore")


# In[ ]:


class CustomRoberta(nn.Module):
    def __init__(self):
        super(CustomRoberta, self).__init__()
        self.num_labels = 1
        self.roberta = transformers.XLMRobertaModel.from_pretrained("xlm-roberta-large", output_hidden_states=False, num_labels=1)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(1024, self.num_labels)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None):

        _, o2 = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)
        
        #apool = torch.mean(o1, 1)
        #mpool, _ = torch.max(o1, 1)
        #cat = torch.cat((apool, mpool), 1)
        #bo = self.dropout(cat)
        logits = self.classifier(o2)       
        outputs = logits
        return outputs


# In[ ]:


df = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/test.csv")


# In[ ]:


tokenizer = transformers.XLMRobertaTokenizer.from_pretrained('xlm-roberta-large', do_lower_case=False)


# In[ ]:


def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    
    return np.array(enc_di['input_ids'])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'test = regular_encode(df.content.values, tokenizer, maxlen=192)')


# In[ ]:


model = CustomRoberta()
model.load_state_dict(torch.load("../input/xlm-roberta-pytorch-xla-tpu/xlm_roberta_model.bin"))
model.eval()


# In[ ]:


test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test))


test_data_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    drop_last=False,
    num_workers=4,
    shuffle=False
)


# In[ ]:


device = xm.xla_device()
with torch.no_grad():
    fin_outputs = []
    model.to(device)
    for bi, d in tqdm(enumerate(test_data_loader),total=len(test_data_loader)):
        ids = d[0]

        ids = ids.to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
        )
        
        #outputs = torch.sigmoid(outputs)
        outputs_np = outputs.cpu().detach().numpy().tolist()
        fin_outputs.extend(outputs_np)


# In[ ]:


fin_outputs = [item for sublist in fin_outputs for item in sublist]
sample = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")
sample.loc[:, "toxic"] = np.array(fin_outputs)
sample.to_csv("submission.csv", index=False)


# In[ ]:


sample.head()

