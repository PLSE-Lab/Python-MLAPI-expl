#!/usr/bin/env python
# coding: utf-8

# ### TPU Imports

# In[ ]:


# !curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
# !python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev


# In[ ]:


import warnings
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

import warnings
warnings.filterwarnings("ignore")


# ### Imports

# In[ ]:


import os
os.environ['XLA_USE_BF16']="1"
os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'

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
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from transformers import AutoTokenizer, AutoModel, AutoConfig
import sys
from sklearn import metrics, model_selection
from fastai.text import *


# ### Dataset

# In[ ]:


INPUT_DATA_PATH = Path("/kaggle/input/xlmrobertabase/xlm_roberta_large_processed/"); INPUT_DATA_PATH.ls()


# In[ ]:


train_inputs = pd.read_pickle(INPUT_DATA_PATH/"translated_inputs.pkl")
valid_inputs = pd.read_pickle(INPUT_DATA_PATH/"valid_inputs.pkl")
test_inputs = pd.read_pickle(INPUT_DATA_PATH/"test_inputs.pkl")


# In[ ]:


class JIGSAWDataset(Dataset):
    def __init__(self, inputs, tokenizer=None, is_test=False, do_tfms:Dict=None, pseudo_inputs=None):

        # eval
        self.inputs = inputs

        # augmentation
        self.is_test = is_test
        self.tokenizer = tokenizer
        self.do_tfms = do_tfms
        self.pseudo_inputs = pseudo_inputs
        if self.pseudo_inputs: self.pseudo_idxs = list(range(len(self.pseudo_inputs)))

    def __getitem__(self, i):
        'fastai requires (xb, yb) to return'

        input_ids = tensor(self.inputs['input_ids'][i])
        attention_mask = tensor(self.inputs['attention_mask'][i])

        if not self.is_test:
            toxic = self.inputs['toxic'][i]

#             if self.do_tfms:
#                 if self.pseudo_inputs and (np.random.uniform() < self.do_tfms["random_replace_with_pseudo"]["p"]):
#                     rand_idx = np.random.choice(self.pseudo_idxs)

#                     input_ids = tensor(self.pseudo_inputs[rand_idx]['input_ids'])
#                     attention_mask = tensor(self.pseudo_inputs[rand_idx]['attention_mask'])
#                     start_position, end_position = self.pseudo_inputs[rand_idx]['start_end_tok_idxs']
#                     start_position, end_position = tensor(start_position), tensor(end_position)

#                 else:
#                     augmentor = TSEDataAugmentor(self.tokenizer,
#                              input_ids,
#                              attention_mask,
#                              start_position, end_position)

#                     if np.random.uniform() < self.do_tfms["random_left_truncate"]["p"]:
#                         augmentor.random_left_truncate()
#                     if np.random.uniform() < self.do_tfms["random_right_truncate"]["p"]:
#                         augmentor.random_right_truncate()
#                     if np.random.uniform() < self.do_tfms["random_replace_with_mask"]["p"]:
#                         augmentor.random_replace_with_mask(self.do_tfms["random_replace_with_mask"]["mask_p"])

#                     input_ids = augmentor.input_ids
#                     attention_mask = augmentor.attention_mask
#                     start_position, end_position = tensor(augmentor.ans_start_pos), tensor(augmentor.ans_end_pos)


        xb = (input_ids, attention_mask)
        if not self.is_test: yb = toxic
        else: yb = 0

        return xb, yb

    def __len__(self): return len(self.inputs)


# In[ ]:


train_ds = JIGSAWDataset(train_inputs)
valid_ds = JIGSAWDataset(valid_inputs)
test_ds = JIGSAWDataset(test_inputs)


# In[ ]:


train_ds[0]


# In[ ]:





# In[ ]:


class JigsawArrayDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids:np.array, attention_mask:np.array, toxic:np.array=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.toxic = toxic
    
    def __getitem__(self, idx):
        xb = (tensor(self.input_ids[idx]), tensor(self.attention_mask[idx]))
        yb = tensor(0.) if self.toxic is None else tensor(self.toxic[idx])
        return xb,yb    
        
    def __len__(self):
        return len(self.input_ids)


# In[ ]:


XLM_PROCESSED_PATH = Path("/kaggle/input/xlmrobertabase/xlm_roberta_processed/"); XLM_PROCESSED_PATH.ls()


# In[ ]:


# # remove eng
# train_input_ids = train_input_ids[train_lang != "en"]
# train_attetion_mask = train_attetion_mask[train_lang != "en"]
# train_toxic = train_toxic[train_lang != "en"]
# train_lang = train_lang[train_lang != "en"]


# In[ ]:


# labels for stratified batch sampler
train_stratify_labels = array(s1+s2 for (s1,s2) in zip(train_lang, train_toxic.astype(str)))
labels2int = {v:k for k,v in enumerate(np.unique(train_stratify_labels))}
labels = [labels2int[o] for o in train_stratify_labels]
balanced_sampler = BalanceClassSampler(labels)


# In[ ]:


labels2int


# In[ ]:


train_ds = JigsawArrayDataset(train_input_ids, train_attetion_mask, train_toxic)


# In[ ]:


# del train_input_ids, train_attetion_mask, train_toxic, train_lang
# del train_stratify_labels, labels2int, labels
# gc.collect()


# In[ ]:


# valid_ds
valid_ds = JigsawArrayDataset(*[np.load(XLM_PROCESSED_PATH/'valid_inputs/input_ids.npy'),
                                np.load(XLM_PROCESSED_PATH/'valid_inputs/attention_mask.npy'),
                                np.load(XLM_PROCESSED_PATH/'valid_inputs/toxic.npy')])


# In[ ]:


len(train_ds), len(valid_ds)


# ### Model

# In[ ]:


def get_xlm_roberta(modelname="xlm-roberta-base"):        
    conf = AutoConfig.from_pretrained(modelname)
    conf.output_hidden_states = True
    model = AutoModel.from_pretrained(modelname, config=conf)
    return model


# In[ ]:


class Head(Module):
    "Concat Pool over sequence"
    def __init__(self, modelname="xlm-roberta-base", p=0.5):
        
        self.d0 = nn.Dropout(p)
        if modelname == "xlm-roberta-base": self.l0 = nn.Linear(768*4, 2)
        elif modelname == "xlm-roberta-large": self.l0 = nn.Linear(1024*4, 2)
        else: raise Exception("Invalid model")
        
    def forward(self, x):
        x = self.d0(x)
        x = torch.cat([x.permute(0,-1,-2).mean(-1), 
                       x.permute(0,-1,-2).max(-1).values], -1)
        x = self.l0(x) 
        return x

class JigsawModel(Module):
    def __init__(self, model, head):
        self.sequence_model = model
        self.head = head

    def forward(self, *xargs):
        inp = {}
        inp["input_ids"] = xargs[0]
        inp["attention_mask"] = xargs[1]
        _, _, hidden_states = self.sequence_model(**inp)
        # feed last 2 hidden states
        x = torch.cat(hidden_states[-2:], -1)
        return self.head(x)


# In[ ]:


modelname = "xlm-roberta-large"
model = get_xlm_roberta(modelname=modelname)
head = Head(modelname=modelname)
jigsaw_model = JigsawModel(model, head)


# In[ ]:


# xb,yb = train_ds[:3]
# out = jigsaw_model(*xb)
# out


# # Training

# In[ ]:


# loss_fn = nn.CrossEntropyLoss()
loss_fn = LabelSmoothingCrossEntropy(0.1)
# def loss_fn(outputs, targets): return loss(outputs, targets)


# In[ ]:


def reduce_fn(vals): return sum(vals) / len(vals)


# In[ ]:


def train_fn(data_loader, model, optimizer, device, num_batches, scheduler=None):
    model.train()
    tk0 = tqdm(data_loader, total=num_batches, desc="Training", disable=not xm.is_master_ordinal())
    for bi, (xb,yb) in enumerate(tk0):

        input_ids, attention_mask = xb
        input_ids = input_ids.to(device, dtype=torch.long)
        attention_mask = attention_mask.to(device, dtype=torch.long)
        yb = yb.to(device, dtype=torch.float)
        
        model.zero_grad()
        out = model(input_ids, attention_mask)
        
        loss = loss_fn(out, yb)
        loss.backward()
        xm.optimizer_step(optimizer)
        scheduler.step()
        print_loss = xm.mesh_reduce('loss_reduce', loss, reduce_fn)
        tk0.set_postfix(loss=print_loss.item())   


# In[ ]:


from sklearn.metrics import roc_auc_score

def eval_fn(data_loader, model, device, num_batches):
    model.eval()
    preds, targs = [], []
   
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=num_batches, desc="Evaluating", disable=not xm.is_master_ordinal())
        for bi, (xb,yb) in enumerate(tk0):

            input_ids, attention_mask = xb
            input_ids = input_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device, dtype=torch.long)
            yb = yb.to(device, dtype=torch.float)
            out = model(input_ids, attention_mask)

            preds.append(to_cpu(out.softmax(-1)[:,1]))
            targs.append(to_cpu(yb))

    return roc_auc_score(torch.cat(targs), torch.cat(preds))


# ### TPULearner

# In[ ]:


from torch.optim.lr_scheduler import *


# In[ ]:


def get_optimizer(model, opt_func=AdamW, lr=1e-5):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias","LayerNorm.weight"]
    optimizer_parameters = [
        {
            'params': [
                p for n, p in param_optimizer if not any(
                    nd in n for nd in no_decay
                )
            ], 
         'weight_decay': 0.001
        },
        {
            'params': [
                p for n, p in param_optimizer if any(
                    nd in n for nd in no_decay
                )
            ], 
            'weight_decay': 0.0
        },
    ]
    optimizer = opt_func(optimizer_parameters, lr=lr)
    return optimizer


# In[ ]:


from catalyst.data.sampler import DistributedSamplerWrapper, BalanceClassSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


# In[ ]:


class TPULearner:
    def __init__(self, model:Module, train_ds:Dataset, valid_ds:Dataset, test_ds:Dataset, opt_func, sched_func=None, sampler=None, bs=128):
        self.model = model
        self.train_ds, self.valid_ds, self.test_ds = train_ds, valid_ds, test_ds
        self.bs = bs
        self.sampler = sampler
        self.opt_func = opt_func
        self.sched_func = sched_func
    
    
    @property
    def device(self): return xm.xla_device()

    @property
    def xmodel(self): return self.model.to(self.device)
    
    @property
    def opt(self): return self.opt_func(self.xmodel)
    
    
    
    
    @property
    def train_dl(self):
        if self.sampler:
            train_sampler = DistributedSamplerWrapper(self.sampler, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True)
        else: 
            train_sampler = DistributedSampler(self.train_ds, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True)
        train_dl = DataLoader(self.train_ds, batch_size=self.bs, sampler=train_sampler, drop_last=True, num_workers=2)
        return train_dl    
    @property
    def train_pl(self): return pl.ParallelLoader(self.train_dl, [self.device]).per_device_loader(self.device)
        
        
    @property
    def valid_dl(self):
        valid_sampler = DistributedSampler(self.valid_ds, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=False)
        valid_dl = DataLoader(self.valid_ds, batch_size=self.bs*2, sampler=valid_sampler,drop_last=False,num_workers=1)
        return valid_dl
    @property
    def valid_pl(self): return pl.ParallelLoader(self.valid_dl, [self.device]).per_device_loader(self.device)

    
    @property
    def test_dl(self): raise NotImplementedError    
    @property
    def test_pl(self): raise NotImplementedError    


# In[ ]:


def train(self, model, opt, scheduler):
    "Train a single epoch with model and opt"
    model.train()
    
    tk0 = tqdm(self.train_pl, total=len(self.train_pl), desc="Training", disable=not xm.is_master_ordinal())
    
    for bi, (xb,yb) in enumerate(tk0):
        
        if not is_listy(xb): xb = listify(xb)
        xb = [x.to(self.device) for x in xb]
        yb = yb.to(self.device)

        model.zero_grad()
        out = model(*xb)

        loss = loss_fn(out, yb)
        loss.backward()
        xm.optimizer_step(opt)
        scheduler.step()
        

        print_loss = xm.mesh_reduce('loss_reduce', loss, reduce_fn)
        tk0.set_postfix(loss=print_loss.item())   


# In[ ]:


from sklearn.metrics import roc_auc_score

def auc(preds, targs):
    return roc_auc_score(targs, preds.softmax(-1)[:,1])

def validate(self, model, eval_func):
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        tk0 = tqdm(self.valid_pl, total=len(self.valid_pl), desc="Evaluating", disable=not xm.is_master_ordinal())
        for bi, (xb,yb) in enumerate(tk0):

            if not is_listy(xb): xb = listify(xb)
            xb = [x.to(self.device) for x in xb]
            yb = yb.to(self.device)
            out = model(*xb)

            preds.append(to_cpu(out))
            targs.append(to_cpu(yb))

    score = eval_func(torch.cat(preds), torch.cat(targs))
    reduced_score = xm.mesh_reduce('reduced_score', score, reduce_fn)
    xm.master_print(f'{eval_func.__name__}={reduced_score}')
    return score


# In[ ]:


def fit(self, epochs=2, eval_func=auc, modelname="mymodel"):
    
    if not hasattr(self, "best_score"): self.best_score = 0    
    opt = self.opt # get optim for the device
    model = self.xmodel # get model for the device
    scheduler = self.sched_func(opt) # get scheduler for the device
    
    
    for i in range(epochs):        
        self.train(model, opt, scheduler)
        score = self.validate(model, eval_func)
       
        if score > self.best_score:
            xm.master_print("Model Improved!!! Saving Model")
            xm.save(model.state_dict(), f"{modelname}.pth")
            self.best_score = score


# In[ ]:


def lr_find(self, start_lr=1e-10, end_lr=10, num_it=200, stop_div = True):
    
    model = self.xmodel # get model for the device
    opt = OptimWrapper(self.opt) # get optim for the device
    sched = Scheduler((start_lr, end_lr), num_it, annealing_exp)
    
    self.losses, self.lrs = tensor([0]), tensor([0])
    
    model.train()
    tk0 = tqdm(self.train_pl, total=len(self.train_pl), desc="Training", disable=not xm.is_master_ordinal())
    for bi, (xb,yb) in enumerate(tk0):

        opt.lr = sched.step()
        
        input_ids, attention_mask = xb
        input_ids = input_ids.to(self.device, dtype=torch.long)
        attention_mask = attention_mask.to(self.device, dtype=torch.long)
        yb = yb.to(self.device, dtype=torch.float)

        model.zero_grad()
        out = model(input_ids, attention_mask)

        loss = loss_fn(out, yb)
        loss.backward()
        xm.optimizer_step(opt)

        print_loss = xm.mesh_reduce('loss_reduce', loss, reduce_fn)
        tk0.set_postfix(loss=print_loss.item())   
        
        if xm.is_master_ordinal():   
            self.losses = torch.cat([self.losses, tensor([print_loss.item()])])
            self.lrs = torch.cat([self.lrs, tensor([opt.lr])])
                        
        if sched.is_done or (torch.isnan(print_loss)): 
            break
    
    xm.master_print("Stopping lr finder...")
    xm.save(self.losses, "lr_find_losses.pt")
    xm.save(self.lrs, "lr_find_lrs.pt")   


# In[ ]:


TPULearner.train = train
TPULearner.validate = validate
TPULearner.fit = fit
TPULearner.lr_find = lr_find


# ### lr_find()
# 
# Uncomment and run the following cells to get the optimal learning rate from the graph.

# In[ ]:


# opt_func = partial(get_optimizer, opt_func=AdamW, lr=5e-5)
# tpu_learner = TPULearner(jigsaw_model, train_ds, valid_ds, None, opt_func, balanced_sampler)

# def _mp_fn(rank, flags):
#     torch.set_default_tensor_type('torch.FloatTensor')
#     tpu_learner.lr_find()

# FLAGS={}
# res = xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')


# In[ ]:


# losses = torch.load("/kaggle/working/lr_find_losses.pt")[1:]
# lrs = torch.load("/kaggle/working/lr_find_lrs.pt")[1:]


# In[ ]:


# smoothen_loss = SmoothenValue(0.98)
# smooth_losses = []
# for l in losses:
#     smoothen_loss.add_value(l)
#     smooth_losses.append(smoothen_loss.smooth)


# In[ ]:


# plt.plot(lrs[10:-30], smooth_losses[10:-30])


# ### fit_one_cycle()

# In[ ]:


# bs = 128
# epochs = 4
# max_lr = 1e-5


# In[ ]:


# total_steps = int(len(balanced_sampler) / bs / 8 * epochs); total_steps


# In[ ]:


# sched_func = partial(OneCycleLR, max_lr=max_lr, total_steps=total_steps)
# tpu_learner = TPULearner(jigsaw_model, train_ds, valid_ds, None, get_optimizer, sched_func, balanced_sampler, bs)


# In[ ]:


# def _mp_fn(rank, flags):
#     torch.set_default_tensor_type('torch.FloatTensor')
#     tpu_learner.fit(epochs, modelname="ft-translated")

# FLAGS={}
# res = xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')


# ### Old Training

# In[ ]:


class config:
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 64
    EPOCHS = 4
    LEARNING_RATE = 1e-4
    MODEL_NAME = "large_finetuned_translated_data"
    TRAINING_DS = train_ds
    VALIDATION_DS = valid_ds


# In[ ]:


def run():

    device = xm.xla_device()
    model = jigsaw_model.to(device)
        
    trn_ds, val_ds = config.TRAINING_DS, config.VALIDATION_DS
    

    train_sampler = DistributedSamplerWrapper(
      balanced_sampler,
      num_replicas=xm.xrt_world_size(),
      rank=xm.get_ordinal(),
      shuffle=True
    )

    train_data_loader = DataLoader(
        trn_ds,
        batch_size=config.TRAIN_BATCH_SIZE,
        sampler=train_sampler,
        drop_last=True,
        num_workers=2
    )

    valid_sampler = DistributedSampler(
      val_ds,
      num_replicas=xm.xrt_world_size(),
      rank=xm.get_ordinal(),
      shuffle=False
    )

    valid_data_loader = DataLoader(
        val_ds,
        batch_size=config.VALID_BATCH_SIZE,
        sampler=valid_sampler,
        drop_last=False,
        num_workers=1
    )

    optimizer = get_optimizer(model, AdamW, lr=config.LEARNING_RATE)

    num_train_steps = int(len(balanced_sampler) / config.TRAIN_BATCH_SIZE / xm.xrt_world_size() * config.EPOCHS)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps*0.15),
        num_training_steps=num_train_steps
    )

    best_auc = 0
    xm.master_print("Training is Starting....")

    for epoch in range(config.EPOCHS):
        para_loader = pl.ParallelLoader(train_data_loader, [device]).per_device_loader(device)
        trn_num_batches = len(para_loader)
        train_fn(
            para_loader, 
            model, 
            optimizer, 
            device,
            trn_num_batches,
            scheduler
        )

        para_loader = pl.ParallelLoader(valid_data_loader, [device]).per_device_loader(device)
        val_num_batches = len(para_loader)
        targs_preds = eval_fn(
            para_loader, 
            model, 
            device,
            val_num_batches
        )
        
        auc = xm.mesh_reduce('auc_reduce', targs_preds, reduce_fn)
        xm.master_print(f'Epoch={epoch}, AUC={auc}')
        if auc > best_auc:
            xm.master_print("Model Improved!!! Saving Model")
            xm.save(model.state_dict(), f"{config.MODEL_NAME}.bin")
            best_auc = auc


# In[ ]:


def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    a = run()

FLAGS={}
xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')


# In[ ]:





# ### fin

# In[ ]:




