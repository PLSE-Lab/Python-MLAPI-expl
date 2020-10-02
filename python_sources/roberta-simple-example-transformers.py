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


get_ipython().system('pip install transformers')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow.keras.backend as K
import gc
import time
import random
import os
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import KFold,StratifiedKFold
from math import floor, ceil
from transformers import AdamW,BertForSequenceClassification


# In[ ]:


sample_submission = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
test = pd.read_csv("../input/google-quest-challenge/test.csv")
train = pd.read_csv("../input/google-quest-challenge/train.csv")

MAX_SEQUENCE_LENGTH = 512


# In[ ]:


print('train shape =', train.shape)
print('test shape =', test.shape)

output_categories = list(train.columns[11:])
input_categories = list(train.columns[[1,2,5]])
print('\noutput categories:\n\t', output_categories)
print('\ninput categories:\n\t', input_categories)


# ## Modified [inital code](https://www.kaggle.com/akensert/bert-base-tf2-0-now-huggingface-transformer) to RoBERTa input format

# In[ ]:


## credit to https://www.kaggle.com/akensert/bert-base-tf2-0-now-huggingface-transformer


def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    first_sep = True
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False 
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


## modified inital code to RoBERTa format
def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [1] * (max_seq_length-len(token_ids))
    return input_ids

## modified inital code to RoBERTa format
def _trim_input(title, question, answer, max_sequence_length, 
                t_max_len=30, q_max_len=238, a_max_len=238):

    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)
    
    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len+q_len+a_len+6) > max_sequence_length:
        
        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len)/2)
            q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
        else:
            t_new_len = t_max_len
      
        if a_max_len > a_len:
            a_new_len = a_len 
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len
            
            
        if t_new_len+a_new_len+q_new_len+6 != max_sequence_length:
            raise ValueError("New sequence length should be %d, but is %d" 
                             % (max_sequence_length, (t_new_len+a_new_len+q_new_len+6)))
        
        if t_len > t_new_len:
            ind1 = floor(t_new_len/2)
            ind2 = ceil(t_new_len/2)
            t = t[:ind1]+t[-ind2:]
        else:
            t = t[:t_new_len]

        if q_len > q_new_len:
            ind1 = floor(q_new_len/2)
            ind2 = ceil(q_new_len/2)
            q = q[:ind1]+q[-ind2:]
        else:
            q = q[:q_new_len]

        if a_len > a_new_len:
            ind1 = floor(a_new_len/2)
            ind2 = ceil(a_new_len/2)
            a = a[:ind1]+a[-ind2:]
        else:
            a = a[:a_new_len]
    
    return t, q, a

def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""
    
    stoken = ['<s>'] + title + ['</s>','</s>'] + question + ['</s>','</s>'] + answer + ['</s>']

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]

def compute_input_arays(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        t, q, a = instance.question_title, instance.question_body, instance.answer

        t, q, a = _trim_input(t, q, a, max_sequence_length)

        ids, masks, segments = _convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
        
    return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# In[ ]:


class TextDataset(torch.utils.data.TensorDataset):

    def __init__(self, x_train, idxs, targets=None):
        self.input_ids = x_train[0][idxs]
        self.input_masks = x_train[1][idxs]
        self.input_segments = x_train[2][idxs]
        self.targets = targets[idxs] if targets is not None else np.zeros((x_train[0].shape[0], 30))

    def __getitem__(self, idx):
#         x_train = self.x_train[idx]
        input_ids =  self.input_ids[idx]
        input_masks = self.input_masks[idx]
        input_segments = self.input_segments[idx]

        target = self.targets[idx]

        return input_ids, input_masks, input_segments, target

    def __len__(self):
        return len(self.input_ids)


# In[ ]:


from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, get_cosine_with_hard_restarts_schedule_with_warmup


# In[ ]:


pretrained_weights = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)


# In[ ]:


x_train = compute_input_arays(train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
y_train = compute_output_arrays(train, output_categories)
x_test = compute_input_arays(test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)


# In[ ]:


bert_config = RobertaConfig.from_pretrained(pretrained_weights) 
bert_config.num_labels = 30


# In[ ]:


NFOLDS = 5
BATCH_SIZE = 4
EPOCHS = 1
SEED = 7345
num_warmup_steps = 100
lr = 3e-5


gradient_accumulation_steps = 1
seed_everything(SEED)

model_list = list()


y_oof = np.zeros((len(train), 30))
test_pred = np.zeros((len(test), 30))


y_oof = np.zeros((len(train), 30))
test_pred = np.zeros((len(test), 30))

kf = KFold(n_splits=NFOLDS, shuffle=True)

test_loader = torch.utils.data.DataLoader(TextDataset(x_test, test.index),batch_size=BATCH_SIZE, shuffle=False)


for i, (train_idx, valid_idx) in enumerate(kf.split(x_train[0])):
    
    
    print(f'fold {i+1}')

    ## loader
    train_loader = torch.utils.data.DataLoader(TextDataset(x_train, train_idx, y_train),batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(TextDataset(x_train, valid_idx, y_train),batch_size=BATCH_SIZE, shuffle=False)
    

    t_total = len(train_loader)//gradient_accumulation_steps*EPOCHS


    net = RobertaForSequenceClassification.from_pretrained(pretrained_weights, config=bert_config)
    net.cuda()
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = AdamW(net.parameters(), lr = lr)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)  # PyTorch scheduler


    for epoch in range(EPOCHS):  

        start_time = time.time()
        avg_loss = 0.0
        net.train()


        for step, data in enumerate(train_loader):

            # get the inputs
            input_ids, input_masks, input_segments, labels = data


            pred = net(input_ids = input_ids.long().cuda(),
                             labels = None,
                             attention_mask = input_masks.cuda(),
                            )[0]
            
            
            loss = loss_fn(pred, labels.cuda())
        
            avg_loss += loss.item()
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:

                # Calling the step function on an Optimizer makes an update to its parameters
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
        avg_val_loss = 0.0

        valid_preds = np.zeros((len(valid_idx), 30))
        true_label = np.zeros((len(valid_idx), 30))
        
        for j,data in enumerate(val_loader):

            # get the inputs
            input_ids, input_masks, input_segments, labels = data
            pred = net(input_ids = input_ids.long().cuda(),
                             labels = None,
                             attention_mask = input_masks.cuda(),
                            )[0]

            loss_val = loss_fn(pred, labels.cuda())
            avg_val_loss += loss_val.item()
            
            pred = torch.sigmoid(pred)
            valid_preds[j * BATCH_SIZE:(j+1) * BATCH_SIZE] = pred.cpu().detach().numpy()
            true_label[j * BATCH_SIZE:(j+1) * BATCH_SIZE]  = labels


        elapsed_time = time.time() - start_time 

        score = 0
        for i in range(30):
          s = np.nan_to_num(
                    spearmanr(true_label[:, i], valid_preds[:, i]).correlation / 30)
          score += s

        

        print('Epoch {}/{} \t loss={:.4f}\t val_loss={:.4f}\t spearmanr={:.4f}\t time={:.2f}s'.format(epoch+1, EPOCHS, avg_loss/len(train_loader),avg_val_loss/len(val_loader),score, elapsed_time))

    model_list.append(net)
    y_oof[valid_idx] = valid_preds


    result = list()
    with torch.no_grad():
        for data in test_loader:
            input_ids, input_masks, input_segments, labels = data
            y_pred = net(input_ids = input_ids.long().cuda(),
                                labels = None,
                                attention_mask = input_masks.cuda(),
                            )[0]

            y_pred = torch.sigmoid(y_pred)
            result.extend(y_pred.cpu().detach().numpy())
            
    test_pred += np.array(result)/NFOLDS


        


# In[ ]:


sample_submission.loc[:, output_categories] = test_pred
sample_submission.to_csv('submission.csv', index=False)


# In[ ]:




