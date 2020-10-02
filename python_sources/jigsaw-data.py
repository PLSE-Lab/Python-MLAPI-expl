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

# Any results you write to the current directory are saved as output.


# In[ ]:


import torch
from scipy import stats

from tqdm import tqdm
from collections import OrderedDict, namedtuple
import torch.nn as nn
from torch.optim import lr_scheduler
import joblib

import logging
import transformers
import sys


# In[ ]:


class BERTBaseUncased(nn.Module):
    def __init__(self, bert_path):
        super(BERTBaseUncased, self).__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768 * 2, 1)

    def forward(
            self,
            ids,
            mask,
            token_type_ids
    ):
        o1, o2 = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids)
        
        apool = torch.mean(o1, 1)
        mpool, _ = torch.max(o1, 1)
        cat = torch.cat((apool, mpool), 1)

        bo = self.bert_drop(cat)
        p2 = self.out(bo)
        return p2


class DatasetTest:
    def __init__(self, comment_text, tokenizer, max_length):
        self.comment_text = comment_text
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, item):
        comment_text = str(self.comment_text[item])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]
        
        padding_length = self.max_length - len(ids)
        
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
        }


# In[ ]:


data = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/test.csv")


# In[ ]:


data


# In[ ]:


tokenizer = transformers.BertTokenizer.from_pretrained("../input/bert-base-multilingual-uncased/", do_lower_case=True)


# In[ ]:


device = "cuda"
model = BERTBaseUncased(bert_path="../input/bert-base-multilingual-uncased/").to(device)
model.load_state_dict(torch.load("../input/jbertml/model.bin"))
model.eval()


# In[ ]:


valid_dataset = DatasetTest(
        comment_text=data.content.values,
        tokenizer=tokenizer,
        max_length=192
)

valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=64,
    drop_last=False,
    num_workers=4,
    shuffle=False
)


# In[ ]:


with torch.no_grad():
    fin_outputs = []
    for bi, d in tqdm(enumerate(valid_data_loader)):
        ids = d["ids"]
        mask = d["mask"]
        token_type_ids = d["token_type_ids"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        outputs_np = outputs.cpu().detach().numpy().tolist()
        fin_outputs.extend(outputs_np)


# In[ ]:


sample = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")
sample.loc[:, "toxic"] = fin_outputs
sample.to_csv("submission.csv", index=False)


# In[ ]:


sample.head()


# In[ ]:




