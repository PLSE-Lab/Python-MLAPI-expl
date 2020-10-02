#!/usr/bin/env python
# coding: utf-8

# - Original Abhisekh's code
# - Data setup from https://www.kaggle.com/xhlulu/jigsaw-tpu-xlm-roberta

# In[ ]:


get_ipython().system('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
get_ipython().system('python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev')


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
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule, XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig
import sys
from sklearn import metrics, model_selection


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
warnings.filterwarnings("ignore")


# In[ ]:


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class BERTDatasetTraining:
    def __init__(self, comment_text, targets, tokenizer, max_length):
        self.comment_text = comment_text
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.targets = targets

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
            truncation_strategy="longest_first",
            pad_to_max_length=True,
        )
        
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[item], dtype=torch.float)
        }


# In[ ]:


class CustomRoberta(nn.Module):
    def __init__(self):
        super(CustomRoberta, self).__init__()
        self.num_labels = 1
        self.roberta = transformers.XLMRobertaModel.from_pretrained("xlm-roberta-base", output_hidden_states=False, num_labels=1)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(768*2, self.num_labels)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None):

        o1, o2 = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)
        
        apool = torch.mean(o1, 1)
        mpool, _ = torch.max(o1, 1)
        cat = torch.cat((apool, mpool), 1)
        bo = self.dropout(cat)
        logits = self.classifier(bo)       
        outputs = logits
        return outputs


# In[ ]:


mx = CustomRoberta();
mx


# In[ ]:


train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv", usecols=["comment_text", "toxic"])
train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv", usecols=["comment_text", "toxic"])
train2.toxic = train2.toxic.round().astype(int)

df_valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')

df_train = pd.concat([
    train1[['comment_text', 'toxic']],
    train2[['comment_text', 'toxic']].query('toxic==1'),
    train2[['comment_text', 'toxic']].query('toxic==0').sample(n=100000, random_state=0)
])

del train1, train2
import gc; gc.collect();

df_train.shape, df_valid.shape


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


tokenizer = transformers.XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'x_train = regular_encode(df_train.comment_text.values, tokenizer, maxlen=128)\nx_valid = regular_encode(df_valid.comment_text.values, tokenizer, maxlen=128)')


# In[ ]:


train_dataset=torch.utils.data.TensorDataset(torch.Tensor(x_train),torch.Tensor(df_train.toxic.values))
valid_dataset=torch.utils.data.TensorDataset(torch.Tensor(x_valid),torch.Tensor(df_valid.toxic.values))


# In[ ]:


def loss_fn(outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


# In[ ]:


def train_loop_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    for bi, d in enumerate(data_loader):

        ids = d[0]
        targets = d[1]

        ids = ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        # input_ids=None, attention_mask=None, position_ids=None, head_mask=None, inputs_embeds=None,
        xm.master_print('model forward now')
        outputs = model(
            input_ids=ids,
        )
        xm.master_print('done forward')
        loss = loss_fn(outputs, targets)
        if bi % 500 == 0:
            xm.master_print(f'bi={bi}, loss={loss}')
        xm.master_print('model backward now')
        loss.backward()
        xm.optimizer_step(optimizer)
        if scheduler is not None:
            scheduler.step()

def eval_loop_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    for bi, d in enumerate(data_loader):
        ids = d[0]
        targets = d[0]

        ids = ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        outputs = model(
            input_ids=ids,
        )

        targets_np = targets.cpu().detach().numpy().tolist()
        outputs_np = outputs.cpu().detach().numpy().tolist()
        fin_targets.extend(targets_np)
        fin_outputs.extend(outputs_np)    

    return fin_outputs, fin_targets


# In[ ]:


def _run():
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 4
    EPOCHS = 2


    train_sampler = torch.utils.data.distributed.DistributedSampler(
          train_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=True)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        sampler=train_sampler,
        drop_last=True,
        num_workers=4,
    )

    valid_sampler = torch.utils.data.distributed.DistributedSampler(
          valid_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=False)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=32,
        sampler=valid_sampler,
        drop_last=False,
        num_workers=4
    )

    device = xm.xla_device()
    model = mx.to(device)
    xm.master_print('done loading model')

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    lr = 2e-5 * xm.xrt_world_size()
    num_train_steps = int(len(train_dataset) / TRAIN_BATCH_SIZE / xm.xrt_world_size() * EPOCHS)
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )
    xm.master_print(f'num_train_steps = {num_train_steps}, world_size={xm.xrt_world_size()}')


    for epoch in range(EPOCHS):
        para_loader = pl.ParallelLoader(train_data_loader, [device])
        xm.master_print('parallel loader created... training now')
        train_loop_fn(para_loader.per_device_loader(device), model, optimizer, device, scheduler=scheduler)

        para_loader = pl.ParallelLoader(valid_data_loader, [device])
        o, t = eval_loop_fn(para_loader.per_device_loader(device), model, device)
        xm.save(model.state_dict(), "xlm_roberta_model.bin")
        auc = metrics.roc_auc_score(np.array(t) >= 0.5, o)
        xm.master_print(f'AUC = {auc}')


# In[ ]:


import os
os.environ['XLA_USE_BF16']="1"


# In[ ]:


# Start training processes
def _mp_fn(rank, flags):
    #torch.set_default_tensor_type('torch.FloatTensor')
    a = _run()

FLAGS={}
xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')

