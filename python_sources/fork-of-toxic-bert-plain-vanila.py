#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/nvidiaapex/repository/NVIDIA-apex-39e153a"))
#print(os.listdir("../input/glove-global-vectors-for-word-representation"))
#print(os.listdir("../input/jigsaw-unintended-bias-in-toxicity-classification"))
#print(os.listdir("../input/fasttext-crawl-300d-2m"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Installing Nvidia Apex
get_ipython().system(' pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ../input/nvidiaapex/repository/NVIDIA-apex-39e153a')


# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import pkg_resources
import seaborn as sns
import time
import scipy.stats as stats
import gc
import re
import operator 
import sys
from sklearn import metrics
from sklearn import model_selection
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
from nltk.stem import PorterStemmer
from sklearn.metrics import roc_auc_score
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm, tqdm_notebook
import os
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings
warnings.filterwarnings(action='once')
import pickle
from apex import amp
import shutil


# In[ ]:


tqdm.pandas()


# In[ ]:


device=torch.device('cuda')


# In[ ]:


MAX_SEQUENCE_LENGTH = 220
SEED = 1234
EPOCHS = 1
Data_dir="../input/jigsaw-unintended-bias-in-toxicity-classification"
Input_dir = "../input"
WORK_DIR = "../working/"
# full_length=1804874
num_to_load=1804800
valid_size= 0                         #Validation Size
# num_to_load=full_length-valid_size                         #Train size to match time limit
# num_to_load=20000                         #Train size to match time limit
# valid_size= 5000                         #Validation Size
TOXICITY_COLUMN = 'target'


# In[ ]:


# Add the Bart Pytorch repo to the PATH
# using files from: https://github.com/huggingface/pytorch-pretrained-BERT
package_dir_a = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"
sys.path.insert(0, package_dir_a)

from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification,BertAdam,OpenAIAdam


# In[ ]:


# Translate model from tensorflow to pytorch
BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'
convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
    BERT_MODEL_PATH + 'bert_model.ckpt',
BERT_MODEL_PATH + 'bert_config.json',
WORK_DIR + 'pytorch_model.bin')

shutil.copyfile(BERT_MODEL_PATH + 'bert_config.json', WORK_DIR + 'bert_config.json')


# In[ ]:


os.listdir("../working")


# In[ ]:


# This is the Bert configuration file
from pytorch_pretrained_bert import BertConfig

bert_config = BertConfig('../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'+'bert_config.json')


# In[ ]:


# Converting the lines to BERT format
# Thanks to https://www.kaggle.com/httpwwwfszyc/bert-in-keras-taming
def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in tqdm_notebook(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    print(longer)
    return np.array(all_tokens)


# In[ ]:


# def convert_line(tl, max_seq_length,tokenizer):
#     max_seq_length -=2
#     tokens_a = tokenizer.tokenize(tl)
#     if len(tokens_a)>max_seq_length:
#         tokens_a = tokens_a[:max_seq_length]
#     one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
#     return one_token


# In[ ]:


BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'


# In[ ]:


get_ipython().run_cell_magic('time', '', 'tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=True)\n\nnp.random.seed(SEED)\nchosen_idx = np.random.choice(num_to_load+valid_size,size = num_to_load+valid_size,replace=False) \ntrain_df = pd.read_csv(os.path.join(Data_dir,"train.csv")).iloc[chosen_idx] \n# train_df = pd.read_csv(os.path.join(Data_dir,"train.csv")).sample(num_to_load+valid_size,random_state=SEED)\nprint(\'loaded %d records\' % len(train_df))\n\n# Make sure all comment_text values are strings\ntrain_df[\'comment_text\'] = train_df[\'comment_text\'].astype(str)\n# train_df[\'comment_text\'] = train_df[\'comment_text\'].progress_apply(lambda x:preprocess(x))\n')


# In[ ]:


# %%time
sequences = convert_lines(train_df["comment_text"].fillna("DUMMY_VALUE"),MAX_SEQUENCE_LENGTH,tokenizer)


# In[ ]:


# %%time
# from joblib import Parallel, delayed
# train_lines = train_df['comment_text'].fillna("DUMMY_VALUE").values.tolist()
# sequences1 = Parallel(n_jobs=2, backend='multiprocessing')(delayed(convert_line)(i, MAX_SEQUENCE_LENGTH, tokenizer) for i in train_lines)


# In[ ]:


# %%time
# import multiprocessing
# from functools import partial
# cpunum = multiprocessing.cpu_count()
# pool = multiprocessing.Pool(cpunum)
# # pool.map(partial(func, b=second_arg), a_args)
# sequence = pool.map(partial(convert_line, max_seq_length=MAX_SEQUENCE_LENGTH,tokenizer=tokenizer),
#                         (i for i in train_df['comment_text'].fillna("DUMMY_VALUE").values.tolist()))


# In[ ]:


train_df=train_df.fillna(0)
# List all identities
identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
x_train = train_df['comment_text']
train_df = train_df.drop(['comment_text'],axis=1)
# convert target to 0,1
train_df['target']=(train_df['target']>=0.5).astype(float)


# In[ ]:



X = sequences[:num_to_load]                
# y = train_df[y_columns].values[:num_to_load]
X_val = sequences[num_to_load:]                
# y_val = train_df[y_columns].values[num_to_load:]


# In[ ]:


# Overall
weights = np.ones((len(train_df),)) / 4
# Subgroup
weights += (train_df[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4
# Background Positive, Subgroup Negative
weights += (( (train_df['target'].values>=0.5).astype(bool).astype(np.int) +
    (train_df[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
# Background Negative, Subgroup Positive
weights += (( (train_df['target'].values<0.5).astype(bool).astype(np.int) +
    (train_df[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
loss_weight = 1.0 / weights.mean()


# In[ ]:


y_columns=['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat','sexual_explicit']
# y_train=np.vstack([train_df['target'],weights]).T
# y_train=np.concatenate((y_train,train_df[y_columns]),axis=1)
y_train=np.array(train_df[y_columns])
y_train = np.hstack((y_train, np.reshape(weights, (-1, 1))))
# y_train = np.hstack((y_train, train_df[y_columns]))


# In[ ]:


y_val = y_train[num_to_load:]
y = y_train[:num_to_load]


# In[ ]:


test_df=train_df.tail(valid_size).copy()
train_df=train_df.head(num_to_load)


# In[ ]:


class SequenceBucketCollator():
    def __init__(self, choose_length, sequence_index, length_index, label_index=None):
        self.choose_length = choose_length
        self.sequence_index = sequence_index
        self.length_index = length_index
        self.label_index = label_index
        
    def __call__(self, batch):
        batch = [torch.stack(x) for x in list(zip(*batch))]
        
        sequences = batch[self.sequence_index]
        lengths = batch[self.length_index]
        
        length = self.choose_length(lengths)
#         print(length)
        mask = torch.arange(start=maxlen, end=0, step=-1) < length
        padded_sequences = sequences[:, mask]
        
        batch[self.sequence_index] = padded_sequences
        
        if self.label_index is not None:
            return [x for i, x in enumerate(batch) if i != self.label_index], batch[self.label_index]
    
        return batch


# In[ ]:


train_dataset = torch.utils.data.TensorDataset(torch.tensor(X,dtype=torch.long),torch.tensor(y,dtype=torch.float))


# In[ ]:


def custom_loss(data, targets):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:,-1:])(data[:,:1],targets[:,:1])
    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:,1:],targets[:,1:-1])
    return (bce_loss_1 * loss_weight) + bce_loss_2


# In[ ]:


# def custom_loss(data, targets):
#     ''' Define custom loss function for weighted BCE on 'target' column '''
#     bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:,1:2])(data[:,:1],targets[:,:1])
#     bce_loss_2 = nn.BCEWithLogitsLoss()(data[:,1:],targets[:,2:])
#     return (bce_loss_1 * loss_weight) + bce_loss_2


# In[ ]:


del train_df, y_train, sequences, X
gc.collect()


# In[ ]:


output_model_file = "bert_pytorch.bin"

lr=2e-5
batch_size = 32
accumulation_steps= 1
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

model = BertForSequenceClassification.from_pretrained("../working",cache_dir=None,num_labels=7)
model.zero_grad()
model = model.to(device)
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
train = train_dataset

num_train_optimization_steps = int(EPOCHS*len(train)/batch_size/accumulation_steps)

optimizer = OpenAIAdam(optimizer_grouped_parameters,
                     lr=lr,
                     warmup=0.05,
                     t_total=num_train_optimization_steps)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
model=model.train()


# In[ ]:


from torch.utils import data
from tqdm import tqdm_notebook as tqdm

class LenMatchBatchSampler(data.BatchSampler):
    def __iter__(self):

        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            count_zeros = torch.sum(self.sampler.data_source[idx][0] == 0)
            count_zeros = int(count_zeros / 64) 
            if len(buckets[count_zeros]) == 0:  buckets[count_zeros] = []

            buckets[count_zeros].append(idx)

            if len(buckets[count_zeros]) == self.batch_size:
                batch = list(buckets[count_zeros])
                yield batch
                yielded += 1
                buckets[count_zeros] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch

        assert len(self) == yielded, "produced an inccorect number of batches. expected %i, but yielded %i" %(len(self), yielded)

def trim_tensors(tsrs):
    max_len = torch.max(torch.sum( (tsrs != 0  ), 1))
    if max_len > 2: 
        tsrs = tsrs[:,:max_len]
    return tsrs


# In[ ]:


tq = tqdm_notebook(range(EPOCHS))
for epoch in tq:
#     train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    ran_sampler = data.RandomSampler(train_dataset)
    len_sampler = LenMatchBatchSampler(ran_sampler, batch_size = 32, drop_last = False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler = len_sampler)
    
    avg_loss = 0.
    avg_accuracy = 0.
    lossf=None
    tk0 = tqdm_notebook(enumerate(train_loader),total=len(train_loader),leave=False)
    optimizer.zero_grad()
    for i,(x_batch, y_batch) in tk0:
#         x_batch=x_batch[0]
#         optimizer.zero_grad()
        x_batch = trim_tensors(x_batch)
#         print(x_batch.shape)
        y_pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
#         loss =  F.binary_cross_entropy_with_logits(y_pred,y_batch.to(device))
#         print(y_pred,y_batch)
        loss = custom_loss(y_pred,y_batch.to(device))
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
            optimizer.step()                            # Now we can do an optimizer step
            optimizer.zero_grad()
        if lossf:
            lossf = 0.98*lossf+0.02*loss.item()
        else:
            lossf = loss.item()
        tk0.set_postfix(loss = lossf)
        avg_loss += loss.item() / len(train_loader)
        avg_accuracy += torch.mean(((torch.sigmoid(y_pred[:,0])>0.5) == (y_batch[:,0]>0.5).to(device)).to(torch.float) ).item()/len(train_loader)
    tq.set_postfix(avg_loss=avg_loss,avg_accuracy=avg_accuracy)

# torch.save(model.state_dict(), output_model_file)


# In[ ]:


# Run validation
# The following 2 lines are not needed but show how to download the model for prediction
# model = BertForSequenceClassification(bert_config,num_labels=len(y_columns))
# model.load_state_dict(torch.load(output_model_file ))
model.to(device)
for param in model.parameters():
    param.requires_grad=False
model.eval()
valid_preds = np.zeros((len(X_val)))
valid = torch.utils.data.TensorDataset(torch.tensor(X_val,dtype=torch.long))
valid_loader = torch.utils.data.DataLoader(valid, batch_size=32, shuffle=False)

tk0 = tqdm_notebook(valid_loader)
for i,(x_batch,)  in enumerate(tk0):
    pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
    valid_preds[i*32:(i+1)*32]=pred[:,0].detach().cpu().squeeze().numpy()


# In[ ]:


# From baseline kernel

def calculate_overall_auc(df, model_name):
    true_labels = df[TOXICITY_COLUMN]>=0.5
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)

def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)

def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)



SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive

def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan

def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]>=0.5]
    return compute_auc((subgroup_examples[label]>=0.5), subgroup_examples[model_name])

def compute_bpsn_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[(df[subgroup]>=0.5) & (df[label]<0.5)]
    non_subgroup_positive_examples = df[(df[subgroup]<0.5) & (df[label]>=0.5)]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label]>=0.5, examples[model_name])

def compute_bnsp_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[(df[subgroup]>=0.5) & (df[label]>=0.5)]
    non_subgroup_negative_examples = df[(df[subgroup]<0.5) & (df[label]<0.5)]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label]>=0.5, examples[model_name])

def compute_bias_metrics_for_model(dataset,
                                   subgroups,
                                   model,
                                   label_col,
                                   include_asegs=False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = {
            'subgroup': subgroup,
            'subgroup_size': len(dataset[dataset[subgroup]>=0.5])
        }
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)


# In[ ]:


MODEL_NAME = 'model1'
test_df[MODEL_NAME]=torch.sigmoid(torch.tensor(valid_preds)).numpy()
TOXICITY_COLUMN = 'target'
bias_metrics_df = compute_bias_metrics_for_model(test_df, identity_columns, MODEL_NAME, 'target')
bias_metrics_df
get_final_metric(bias_metrics_df, calculate_overall_auc(test_df, MODEL_NAME))


# In[ ]:


state = {'state_dict': model.state_dict()}
# ,'optimizer': optimizer.state_dict()}
torch.save(state, 'bert_checkpoint_part1.bin')

#model, optimizer, start_epoch, losslogger = load_checkpoint(model, optimizer, losslogger)
#model = model.to(device)
## now individually transfer the optimizer parts...
#for state in optimizer.state.values():
#    for k, v in state.items():
#        if isinstance(v, torch.Tensor):
#            state[k] = v.to(device)


# In[ ]:




