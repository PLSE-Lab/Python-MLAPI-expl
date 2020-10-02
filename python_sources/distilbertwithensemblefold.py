#!/usr/bin/env python
# coding: utf-8

# # 1. Install needed libraries

# In[ ]:


get_ipython().system('pip install ../input/sacremoses/sacremoses-master/ > /dev/null')


# In[ ]:


get_ipython().system('pip install /kaggle/input/transformers/transformers-2.2.1-py3-none-any.whl')


# In[ ]:


get_ipython().system('pip install transformers')


# # 2. load data

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


df_submit = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
df_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
df_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
df_train.text = df_train.text.astype(str)
df_test.text = df_test.text.astype(str)


# In[ ]:


df_train


# In[ ]:


import os
import sys
import glob
import torch
import time
from tqdm import tqdm_notebook as tqdm
import transformers
# DEVICE = torch.device("cuda")
tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
# tokenizer = transformers.DistilBertTokenizer.from_pretrained("../input/distilbertbaseuncased/")
# model = transformers.DistilBertModel.from_pretrained("../input/distilbertbaseuncased/")
# model.to(DEVICE)


# # 3. Preprocess the data for training

# In[ ]:


### SINCE WHOLE USEFUL VECTORS ARE 125, we can just use 129...
### [CLS] + text[82] + [SEP] + location[30] + [SEP] + keyword[7] + [SEP]

def to_ids(tokenizer, text):
    ids = []
    tokens = tokenizer.tokenize(text)
    for token in tokens:
        ids.append(tokenizer._convert_token_to_id(token))
    return ids
def pad_to_length(ids, length):
    ids = ids[:length]
    if len(ids) < length:
        ids += ([0] * (length - len(ids)))
    return ids
def join_segments(tokenizer, *segments):
    vector = [tokenizer._convert_token_to_id('[CLS]')]
    sep_id = tokenizer._convert_token_to_id('[SEP]')
    for segment in segments:
        vector += segment
        vector.append(sep_id)
    return vector


# In[ ]:


train_vectors = []
for _, (text, location, keyword) in tqdm(df_train[['text', 'location', 'keyword']].fillna("-").iterrows(), total=df_train.shape[0]):
    text_ids = pad_to_length(to_ids(tokenizer, text), 85)
    location_ids = pad_to_length(to_ids(tokenizer, location), 31)
    keyword_ids = pad_to_length(to_ids(tokenizer, keyword), 8)
    train_vectors.append(join_segments(tokenizer, text_ids, location_ids, keyword_ids))
X_train = np.array(train_vectors)


# In[ ]:


test_vectors = []
for _, (text, location, keyword) in tqdm(df_test[['text', 'location', 'keyword']].fillna("-").iterrows(), total=df_test.shape[0]):
    text_ids = pad_to_length(to_ids(tokenizer, text), 85)
    location_ids = pad_to_length(to_ids(tokenizer, location), 31)
    keyword_ids = pad_to_length(to_ids(tokenizer, keyword), 8)
    test_vectors.append(join_segments(tokenizer, text_ids, location_ids, keyword_ids))
X_test = np.array(test_vectors)


# In[ ]:


y_train = df_train.target.values


# ## See if there is a severe class imbalance...

# In[ ]:


df_train.target.plot.hist()


# # 4. Train with customized DistilBert classifiers per each fold
#  - After training and validating with each fold, use each classifier to calculate probabilities

# In[ ]:


from transformers import DistilBertForSequenceClassification, DistilBertModel


# In[ ]:


from torch.nn import BCEWithLogitsLoss
from torch import nn
from tqdm.notebook import tqdm
from tqdm import trange
from transformers import AdamW
from torch.optim import Adam
from sklearn.metrics import f1_score, precision_score, recall_score
import torch.nn.functional as F
def train(model, num_epochs, train_dataloader, valid_dataloader, crit_function=nn.CrossEntropyLoss(), device='cpu'):
#     optimizer = AdamW(model.parameters(), lr=3.5e-5, weight_decay=0.01, correct_bias=False)
#     model.to(device)
    """
    Train the model and save the model with the lowest validation loss
    """
#     crit_function = nn.BCEWithLogitsLoss()
    model.to(device)
    start_epoch = 0
#     optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01, correct_bias=False)
    optimizer = Adam(model.parameters(), lr=1.5e-5, weight_decay=0.00)

#     optimizer = torch.optim.Adamax(model.parameters(), lr=3e-5)
    # trange is a tqdm wrapper around the normal python range
    for i in trange(num_epochs, desc="Epoch"):
        # if continue training from saved model
        actual_epoch = start_epoch + i
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0
        num_train_samples = 0

        t = tqdm(total=len(train_data), desc="Training: ", position=0)
        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss = model(b_input_ids, labels=b_labels)
            # store train loss
#             print(loss)
            loss = loss[0]
#             input()
            tr_loss += loss.item()
            num_train_samples += b_labels.size(0)
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            #scheduler.step()
            t.update(n=b_input_ids.shape[0])
        t.close()
        # Update tracking variables
        epoch_train_loss = tr_loss/num_train_samples*batch_size
#         train_loss_set.append(epoch_train_loss)

        print("Train loss: {}".format(epoch_train_loss))

        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables 
        eval_loss = 0
        num_eval_samples = 0

        v_preds = []
        v_labels = []

        # Evaluate data for one epoch
        t = tqdm(total=len(validation_data), desc="Validating: ", position=0)
        eval_losses = []
        for batch in valid_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids,b_labels = batch
            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate validation loss
                logits = model(b_input_ids)[0]
#                 print(b_labels)
#                 print(logits)
                loss = crit_function(logits.view(-1, 2),b_labels.view(-1))
#                 print(loss)
                preds = torch.argmax(F.softmax(logits.view(-1, 2), dim=-1), dim=-1)
#                 print(preds)
#                 input()
                v_labels.append(b_labels.cpu().numpy())
                v_preds.append(preds.cpu().numpy())
                # store valid loss
                eval_losses.append(loss.item())
                num_eval_samples += b_labels.size(0)
            t.update(n=b_labels.shape[0])
        t.close()

        v_labels = np.hstack(v_labels)
        v_preds = np.hstack(v_preds)
        print(v_labels.shape)
        epoch_eval_loss = sum(eval_losses)/len(eval_losses)
#         valid_loss_set.append(epoch_eval_loss)
        print('Epoch #{} Validation Results:'.format(i + 1))
        print('\tvalidation BCE loss: ~{}'.format(epoch_eval_loss))
        print('\tF1: {}'.format(f1_score(v_labels, v_preds)))
        print('\tPrecision: {}'.format(precision_score(v_labels, v_preds)))
        print('\tRecall: {}'.format(recall_score(v_labels, v_preds)))
        print("\n")

    return model


# In[ ]:


def get_pred_probs(model, test_dataloader, test_length ,device):
    model.eval()
    model.to(device)

    # Tracking variables 
    eval_loss = 0
    num_eval_samples = 0

    v_preds = []
    v_probs = []

    # Evaluate data for one epoch
    t = tqdm(total=test_length, desc="Inferencing test data: ", position=0)
    eval_losses = []
    for batch in test_dataloader:
        # Add batch to GPU
#         print(batch)
#         input('PAUSED')
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids= batch[0]
        with torch.no_grad():
            # Forward pass, calculate validation loss
            logits = model(b_input_ids)[0]
            probs = F.softmax(logits.view(-1, 2), dim=-1)
            preds = torch.argmax(probs, dim=-1)
            v_probs.append(probs.cpu().numpy())
            v_preds.append(preds.cpu().numpy())
        t.update(n=probs.shape[0])
    t.close()
    probs = np.vstack(v_probs)
    classifications = np.hstack(v_preds)
    
    return probs, classifications


# In[ ]:


import logging
import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from transformers.modeling_distilbert import DistilBertPreTrainedModel
from transformers.modeling_bert import BertPreTrainedModel
from transformers.file_utils import add_start_docstrings
from transformers.modeling_utils import PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits, PreTrainedModel, SequenceSummary


# ### Credit: HuggingFace @ Github

# In[ ]:


class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
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

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):
        super(DistilBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None):
        distilbert_output = self.distilbert(
            input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
#         pooled_output = torch.mean(hidden_state, 1)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.SELU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)

        outputs = (logits,) + distilbert_output[1:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


# In[ ]:


import gc
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, DataLoader
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
batch_size = 16
kf = StratifiedKFold(n_splits=7, random_state=8980, shuffle=True)
kf.get_n_splits(X_train, y_train)
probs_from_folds = []
for ind, (tr, val) in enumerate(kf.split(X_train, y_train)):
    # new model per split
    print("FOLD #{} STARTING!".format(ind + 1))
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    
#     model = DistilBertForSequenceClassification.from_pretrained("../input/distilbertbaseuncased/")
    
    # split train datatset
    X_tr = X_train[tr]
    y_tr = y_train[tr]
    X_val = X_train[val]
    y_val = y_train[val]
    
    # convert train dataset
    X_tr = torch.tensor(X_tr)
    X_val = torch.tensor(X_val)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    
    # prepare train dataloader
    train_data = TensorDataset(X_tr, y_tr)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,                                  sampler=train_sampler,                                  batch_size=batch_size)

    validation_data = TensorDataset(X_val, y_val)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data,                                       sampler=validation_sampler,                                       batch_size=batch_size)
    
    #start training!
    model = train(model=model, num_epochs=2,                  train_dataloader=train_dataloader, valid_dataloader=validation_dataloader,
                  device='cuda')
    
    
    # prepare test data
    test_data = torch.tensor(X_test)
    test_data = TensorDataset(test_data)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data,                                       sampler=test_sampler,                                       batch_size=batch_size)
    
    probs, _ = get_pred_probs(model, test_dataloader,                   test_length=df_test.shape[0] ,device='cuda')
    probs_from_folds.append(probs)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    
    


# # 5. Calculate mean probabilities from accumulated predictions

# In[ ]:


pff = np.array(probs_from_folds)
pff_mean = np.mean(pff, axis=0)


# In[ ]:


pff_mean


# # 6. Convert softmaxed probs to class

# In[ ]:


preds = np.argmax(pff_mean, axis=1)


# # 7. SUBMIT!

# In[ ]:


df_submit['target'] = preds


# In[ ]:


df_submit


# In[ ]:


df_submit.to_csv('submission.csv', index=False)


# [SUBMISSION FILE](./submission.csv)
