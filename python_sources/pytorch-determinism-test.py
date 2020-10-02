#!/usr/bin/env python
# coding: utf-8

# In[ ]:


FOLD = 0

import os
import time
import math
import requests
import glob

import ast

import numpy as np
import pandas as pd

import mlcrate as mlc

import os

import cv2

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score, f1_score

from skimage.transform import resize

from PIL import Image, ImageDraw

from tqdm import tqdm, tqdm_notebook

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
import torch.utils.checkpoint as checkpoint

import torchvision
from torchvision import transforms, utils

import torchtext
import torchtext.data as data

from torch.nn.utils.rnn import pad_sequence

import spacy
from spacy.lang.en import English

SEED = 1337

NOTIFY_EACH_EPOCH = False

WORKERS = 0
BATCH_SIZE = 512

N_SPLITS = 10

np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

# from https://github.com/floydhub/save-and-resume
def save_checkpoint(state):
    """Save checkpoint if a new best is achieved"""
    print (" Saving checkpoint")

    filename = f'./checkpoint-{state["epoch"]}.pt.tar'
    torch.save(state, filename)
    
def initialize(model, path=None, optimizer=None):   
    if path == None:
        checkpoints = glob.glob('./*.pt.tar')
        path = checkpoints[np.argmax([int(checkpoint.split('checkpoint-')[1].split('.')[0]) for checkpoint in checkpoints])]

    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model'])

    print(f' Loaded checkpoint {path} | Trained for {checkpoint["epoch"] + 1} epochs')
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
          
        epoch = checkpoint['epoch'] + 1
        train_iteration = checkpoint['train_iteration']
        val_iteration = checkpoint['val_iteration']

        return model, optimizer, epoch, train_iteration, val_iteration
    else:
        return model


# In[ ]:


sample_submission = pd.read_csv('../input/sample_submission.csv')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

train_idx, val_idx = list(kfold.split(train))[FOLD]
x_train, x_val = train.iloc[train_idx], train.iloc[val_idx]

x_train.to_csv('train.csv')
x_val.to_csv('val.csv')


# In[ ]:


nlp = English()
def tokenize(sentence):
    x = nlp(sentence)
    return [token.text for token in x]


# In[ ]:


# from http://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext
question_field = data.Field(tokenize=tokenize, lower=True, batch_first=True, include_lengths=True)
target_field = data.Field(sequential=False, use_vocab=False, batch_first=True)

train_fields = [
    ('id', None),
    ('qid', None),
    ('question_text', question_field),
    ('target', target_field)
]

test_fields = [
    ('qid', None),
    ('question_text', question_field)
]

train_dataset, val_dataset = data.TabularDataset.splits('./', train='train.csv', validation='val.csv', format='CSV', skip_header=True, fields=train_fields)
test_dataset = data.TabularDataset('../input/test.csv', format='CSV', skip_header=True, fields=test_fields)

vectors = torchtext.vocab.Vectors('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')#, max_vectors=1000)

question_field.build_vocab(train_dataset, max_size=95000)
question_field.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
pretrained_embedding = question_field.vocab.vectors

train_dataloader, val_dataloader = data.BucketIterator.splits((train_dataset, val_dataset), (BATCH_SIZE, BATCH_SIZE), sort_key=lambda x: len(x.question_text), sort_within_batch=True)
test_dataloader = data.BucketIterator(test_dataset, 1, sort=False, shuffle=False)

print(f'Train Dataset: {len(train_dataset)}')
print(f'Val Dataset: {len(val_dataset)}')
print(f'Test Dataset: {len(test_dataset)}')


# In[ ]:


len(question_field.vocab.itos)


# In[ ]:


# from https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/4
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(SelfAttention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]
            
        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1) # (batch_size, hidden_size, 1)
                            )
    
        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)

        # create mask based on the sentence lengths
        mask = torch.ones(attentions.size(), requires_grad=True).cuda()
        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row
        
        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions

class BaselineLSTM(nn.Module):
    def __init__(self, embedding):
        super(BaselineLSTM, self).__init__()
                
        self.embedding = nn.Embedding.from_pretrained(embedding)
        
        self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        
        self.attention = SelfAttention(128*2, batch_first=True)
        
        self.fc = nn.Linear(128*2, 1)
        self.logit = nn.Linear(1, 1)

    def forward(self,x, x_len):
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True)

        out, (hidden, _) = self.lstm(x)
        
        x, lengths = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        
        x, _ = self.attention(x, lengths) 
        
        x = self.fc(x)
        x = self.logit(x).view(-1)
        
        return x


# In[ ]:


start_epoch = 0
epochs = 7
early_stopping = 10

train_iteration = 0
val_iteration = 0

threshold = 0.35

model = BaselineLSTM(pretrained_embedding).to(device)

optimizer = optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

criterion = nn.BCEWithLogitsLoss()

get_n_params(model)


# In[ ]:


best_train_loss = 1e10
best_val_loss = 1e10

best_train_f1 = 0
best_val_f1 = 0

best_epoch = 0

timer = mlc.time.Timer()
logger = mlc.LinewiseCSVWriter('train_log.csv', header=['epoch', 'lr', 'train_loss', 'val_loss', 'train_f1', 'val_f1'])

for epoch in range(start_epoch, epochs):
    print(f'\n Starting Epoch {epoch} | LR: {optimizer.param_groups[0]["lr"]}')
        
    train_loss = 0
    val_loss = 0

    y_train = []
    train_preds = []
    
    timer.add(epoch)

    model.train()
    for i, batch in tqdm_notebook(enumerate(train_dataloader), total=(int(len(train_dataset) / BATCH_SIZE))):
        (question, length), label = batch.question_text, batch.target.to(device).float()
        question = question.to(device)
        
        out = model(question, length)

        loss = criterion(out, label)

        train_loss += loss.item()

        optimizer.zero_grad()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()
        
        y_train.append(label.detach())
        train_preds.append(out.detach())
            
        train_iteration += 1
          
    model.eval()
    with torch.no_grad():
        
        y_val = []
        val_preds = []

        for j, batch in tqdm_notebook(enumerate(val_dataloader), total=(int(len(val_dataset) / BATCH_SIZE))):
            (question, length), label = batch.question_text, batch.target.to(device).float()
            question = question.to(device)

            out = model(question, length)
            
            loss = criterion(out, label)

            val_loss += loss.item()

            optimizer.zero_grad()
            
            y_val.append(label.detach())
            val_preds.append(out.detach())

            val_iteration += 1
    
    train_loss /= (i + 1)
    val_loss /= (j + 1)

    y_train = torch.cat(y_train, dim=0).reshape(-1, 1)
    y_val = torch.cat(y_val, dim=0).reshape(-1, 1)

    train_preds = torch.cat(train_preds, dim=0).reshape(-1, 1)
    val_preds = torch.cat(val_preds, dim=0).reshape(-1, 1)
    
    train_f1 = f1_score(y_train, (train_preds > threshold))
    val_f1 = f1_score(y_val, (val_preds > threshold))
    
    logger.write([epoch, optimizer.param_groups[0]['lr'], train_loss, val_loss, train_f1, val_f1])
            
    print(f' {timer.fsince(epoch)} | End of Epoch {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss} | Train F1: {round(train_f1, 4)} | Val F1: {round(val_f1, 4)}')
          
    scheduler.step(val_loss)
          
    if val_loss < best_val_loss:
        best_epoch = epoch

        best_train_loss = train_loss
        best_val_loss = val_loss

        best_train_f1 = train_f1
        best_val_f1 = val_f1
          
        save_checkpoint({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'train_iteration': train_iteration,
            'val_iteration': val_iteration
        })

    elif epoch - best_epoch > early_stopping:
        print(f' Val loss has not decreased for {early_stopping} epochs, stopping training')
        break


# In[ ]:


print(f' Training Finished | Best Epoch {best_epoch} | LR: {optimizer.param_groups[0]["lr"]} \n Best Train Loss: {best_train_loss} | Best Val Loss: {best_val_loss} | Best Train F1: {round(best_train_f1, 4)} | Best Val F1: {round(best_val_f1, 4)}')


# In[ ]:


log = pd.read_csv('train_log.csv')
plt.plot(log['epoch'], log['train_loss'], log['val_loss'])
plt.show()
plt.plot(log['epoch'], log['train_f1'], log['val_f1'])


# In[ ]:


thresholds = np.arange(0.1, 0.501, 0.01)

val_scores = []
for threshold in thresholds:
    threshold = np.round(threshold, 2)
    f1 = f1_score(y_val.cpu().numpy(), (torch.sigmoid(val_preds).cpu().numpy() > threshold).astype(int))
    val_scores.append(f1)

best_val_f1 = np.max(val_scores)
best_threshold = np.round(thresholds[np.argmax(val_scores)], 2)

plt.plot(thresholds, val_scores)

print(f' Best threshold: {best_threshold} | Best Train F1: {f1_score(y_train, (torch.sigmoid(train_preds).cpu().numpy() > best_threshold).astype(int))} | Best Val F1: {best_val_f1}')


# In[ ]:


model = BaselineLSTM(pretrained_embedding).to(device)
model = initialize(model)

preds = []

model.eval()
with torch.no_grad():
    for i, batch in tqdm(enumerate(test_dataloader), total=int(len(test_dataset) / BATCH_SIZE)):
        (question, length) = batch.question_text
        question = question.to(device)
        
        out = model(question, length)
        out = torch.sigmoid(out)
        pred = out.detach().cpu().numpy()
        preds.append(pred)
        
preds = np.concatenate(preds, axis=0).reshape(-1, 1)
preds = (preds > best_threshold).astype(int)


# In[ ]:


sample_submission['prediction'] = preds
mlc.kaggle.save_sub(sample_submission, 'submission.csv')


# In[ ]:


sample_submission.head()


# In[ ]:









