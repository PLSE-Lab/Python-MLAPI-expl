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


import gc
import re
import os
import sys
import time
import pickle
import random
import unidecode
from tqdm import tqdm
tqdm.pandas()
from scipy.stats import spearmanr
from gensim.models import Word2Vec
from flashtext import KeywordProcessor
from keras.preprocessing import text, sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, KFold
from transformers import (
    BertTokenizer, BertModel, BertForSequenceClassification, BertConfig,
    WEIGHTS_NAME, CONFIG_NAME, AdamW, get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup,
)

from math import floor, ceil


# In[ ]:


### just get 50,000 training data
train = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv', usecols=['id', 'comment_text', 'toxic']).sample(n=25000)
train_2 = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv', usecols=['id', 'comment_text', 'toxic', 'rating'])
train_2 = train_2.loc[train_2['rating']=='approved'].drop(columns=['rating']).sample(n=25000)

train = pd.concat([train, train_2]).reset_index(drop=True)
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
validation = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')


# In[ ]:


del train_2
gc.collect()


# In[ ]:


tokenizer = BertTokenizer.from_pretrained("/kaggle/input/bert-mmm/bert-base-multilingual-uncased-vocab.txt")


# In[ ]:


def clean_newline(context):
    return re.sub(r'(\n)', ' ', context)

train['clean_text'] = train['comment_text'].apply(lambda x: clean_newline(x))
test['clean_text'] = test['content'].apply(lambda x: clean_newline(x))
validation['clean_text'] = validation['comment_text'].apply(lambda x: clean_newline(x))


# In[ ]:


train['comment_text_len'] = train.clean_text.apply(lambda x : len(x.split()))
test['comment_text_len'] = test.clean_text.apply(lambda x : len(x.split()))
validation['comment_text_len'] = train.clean_text.apply(lambda x : len(x.split()))


# In[ ]:


train.hist('comment_text_len')


# In[ ]:


test.hist('comment_text_len')


# In[ ]:


validation.hist('comment_text_len')


# In[ ]:


MAX_SEQ_LENGTH = 200


# In[ ]:


class TextDataset(torch.utils.data.TensorDataset):

    def __init__(self, data, idxs, targets=None):
        self.input_ids = data[0][idxs]
        self.input_masks = data[1][idxs]
        self.input_segments = data[2][idxs]
        self.targets = targets[idxs] if targets is not None else np.zeros((data[0].shape[0], 1))
    def __getitem__(self, idx):
        input_ids =  self.input_ids[idx]
        input_masks = self.input_masks[idx]
        input_segments = self.input_segments[idx]

        target = self.targets[idx]

        return input_ids, input_masks, input_segments, target

    def __len__(self):
        return len(self.input_ids)


# In[ ]:


def get_bert_tokenize(df):
    
    token_ids, segment_ids, attention_mask = np.zeros((len(df), MAX_SEQ_LENGTH)), np.zeros((len(df), MAX_SEQ_LENGTH)), np.zeros((len(df), MAX_SEQ_LENGTH))
    
    for i, content in tqdm(enumerate(df.clean_text.values), total=len(df)):
        
        content = tokenizer.tokenize(content)[:MAX_SEQ_LENGTH-2]
        token_id = tokenizer.encode(content)
        
        token_ids[i] = token_id + [0] * (MAX_SEQ_LENGTH-len(token_id))
        segment_ids[i] = [0] * MAX_SEQ_LENGTH               
        attention_mask[i] = [1] * len(token_id) + [0] * (MAX_SEQ_LENGTH - len(token_id))
        
    return token_ids, segment_ids, attention_mask


# In[ ]:


get_ipython().run_cell_magic('time', '', 'x_train = get_bert_tokenize(train)\nx_test = get_bert_tokenize(test)\nx_validation = get_bert_tokenize(validation)')


# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# In[ ]:


bert_config = BertConfig.from_pretrained('/kaggle/input/bert-pytorch/bert-base-multilingual-uncased-config.json') 
bert_config.num_labels = 1


# In[ ]:


import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel,BertModel


class CustomizedBert(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):
        super(CustomizedBert, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)

        self.init_weights()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        ## mean max pooling and concatenate to a vector
        
        avg_pool = torch.mean(outputs[0], 1)
        max_pool, _ = torch.max(outputs[0], 1)
        pooled_output = torch.cat((max_pool, avg_pool), 1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # (loss), logits, (hidden_states), (attentions)


class callback:
    def __init__(self):
        self.score = list()
        self.model = list()
    
    def put(self, model, score):
        self.score.append(score)
        self.model.append(model)

    def get_model(self):
        ind = np.argmin(self.score)
        return self.model[ind]
    


# In[ ]:


def model_validation(net, val_loader, validation_length):
    
    avg_val_loss = 0.0
    
    net.eval()
    valid_preds = np.zeros((validation_length, 1))
    true_label = np.zeros((validation_length, 1))
    
    for j, data in enumerate(val_loader):

        # get the inputs
        input_ids, input_masks, input_segments, labels = data
        pred = net(input_ids = input_ids.long().cuda(),
                         labels = None,
                         attention_mask = input_masks.cuda(),
                         token_type_ids = input_segments.long().cuda()
                        )[0]

        loss_val = loss_fn(pred, labels.float().cuda())
        avg_val_loss += loss_val.item()

        valid_preds[j * BATCH_SIZE:(j+1) * BATCH_SIZE] = torch.sigmoid(pred).cpu().detach().numpy()
        true_label[j * BATCH_SIZE:(j+1) * BATCH_SIZE]  = labels.float()


    score = roc_auc_score(true_label, valid_preds)
    
    return valid_preds, avg_val_loss, score


# In[ ]:


from sklearn.metrics import roc_auc_score, accuracy_score

cb = callback()
    
BATCH_SIZE = 32
EPOCHS = 2

SEED = 2020
lr = 3e-5


gradient_accumulation_steps = 1
seed_everything(SEED)

model_list = list()

y_train = train.toxic.values.reshape(-1, 1)
y_val = validation.toxic.values.reshape(-1, 1)

test_pred = np.zeros((len(test), 1))

test_loader = torch.utils.data.DataLoader(TextDataset(x_test, test.index),batch_size=BATCH_SIZE, shuffle=False)

    
    
## loader
train_loader = torch.utils.data.DataLoader(TextDataset(x_train, train.index, y_train),batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(TextDataset(x_validation, validation.index, y_val),batch_size=BATCH_SIZE, shuffle=False)

t_total = len(train_loader)//gradient_accumulation_steps*EPOCHS
warmup_proportion = 0.01
num_warmup_steps = t_total * warmup_proportion

net = CustomizedBert.from_pretrained('/kaggle/input/bert-pytorch/bert-base-multilingual-uncased-pytorch_model.bin', config=bert_config)
net.cuda()

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = AdamW(net.parameters(), lr = lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)  # PyTorch scheduler

print('Start fine tuning BERT')
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
                         token_type_ids = input_segments.long().cuda()
                        )[0]


        loss = loss_fn(pred, labels.float().cuda())

        avg_loss += loss.item()
        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


    valid_preds, val_loss, val_score  = model_validation(net, val_loader, len(validation))

    elapsed_time = time.time() - start_time 


    print('Epoch {}/{} \t loss={:.4f}\t val_loss={:.4f} \t val_score={:.4f}\t time={:.2f}s'
          .format(epoch+1, EPOCHS, avg_loss/len(train_loader), val_loss/len(val_loader), val_score , elapsed_time))

    cb.put(net, val_loss/len(val_loader))




# ## And Lastly, I add validation dataset to fine tuning model finally.

# In[ ]:


net = cb.get_model()

for epoch in range(EPOCHS):  

    start_time = time.time()
    avg_loss = 0.0
    net.train()

    for step, data in enumerate(val_loader):

        # get the inputs
        input_ids, input_masks, input_segments, labels = data


        pred = net(input_ids = input_ids.long().cuda(),
                         labels = None,
                         attention_mask = input_masks.cuda(),
                         token_type_ids = input_segments.long().cuda()
                        )[0]


        loss = loss_fn(pred, labels.float().cuda())

        avg_loss += loss.item()
        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    elapsed_time = time.time() - start_time 
    print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'
          .format(epoch+1, EPOCHS, avg_loss/len(val_loader) elapsed_time))


# In[ ]:


pickle.dump(cb.get_model(), open('bert_model.pkl', 'wb'))


# In[ ]:


test_loader = torch.utils.data.DataLoader(TextDataset(x_test, test.index), batch_size=128, shuffle=False)

result = list()

# net = cb.get_model()
net.eval()
with torch.no_grad():
    for data in test_loader:
        input_ids, input_masks, input_segments, _ = data
        y_pred = net(input_ids = input_ids.long().cuda(),
                            labels = None,
                            attention_mask = input_masks.cuda(),
                            token_type_ids = input_segments.long().cuda(),
                        )[0]
        result.extend(torch.sigmoid(y_pred).cpu().detach().numpy())


# In[ ]:


sub['toxic'] = np.array(result)
sub.to_csv('submission.csv', index=False)
sub.head()


# In[ ]:


sub.hist('toxic')


# In[ ]:


sub.describe()

