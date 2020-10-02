#!/usr/bin/env python
# coding: utf-8

# # Starter kernel based on Abhishek's and akensert's work in pytorch
# 
# ## You can train having internet on and make an inference kernel. 
# 
# Sorry for not committing properly since I need GPU hours, running 3 competitions with GPU/TPU is straining on my resources :)
# 

# In[ ]:


### with no internet
# !pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ../input/nvidia-apex/apex 
### with internet
get_ipython().system('git clone https://github.com/NVIDIA/apex')
get_ipython().system('pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex')


# In[ ]:


import os
import re
import string
import random
import numpy as np
import pandas as pd
import transformers
from transformers import *
import tokenizers
from tqdm.autonotebook import tqdm
from apex import amp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

SEED = 14
def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
seed_all(SEED)


# In[ ]:


max_len = 112 # No tweet is longer than 108
train_batch_size = 64
valid_batch_size = 16
epochs = 5
bert_path = '../input/bert-base-uncased/vocab.txt'
modle_path = "bertbaseuncased.bin"
train_file = "../input/tweet-sentiment-extraction/train.csv"
test_file = "../input/tweet-sentiment-extraction/test.csv"
tokenizer = tokenizers.BertWordPieceTokenizer(bert_path,lowercase=True)


# In[ ]:


class AverageMeter:
    """Computes and stores the average and current value"""
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

def loss_fn(o1, o2, t1, t2):
    # start_logits, end_logits, target_start, target_end
    l1 = nn.BCEWithLogitsLoss()(o1, t1)
    l2 = nn.BCEWithLogitsLoss()(o2, t2)
    return l1 + l2

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):

        tweet = self.tweet[item]
        selected_text = self.selected_text[item]
    
        len_sel_text = len(selected_text)
        idx0 = -1
        idx1 = -1

        for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
            if tweet[ind:ind+len_sel_text] == selected_text:
                idx0 = ind
                idx1 = ind + len_sel_text - 1
                break
        
        char_targets = [0] * len(tweet)
        if idx0 != -1 and idx1 != -1:
            for j in range(idx0, idx1 + 1):
                if tweet[j] != " ":
                    char_targets[j] = 1

        tok_tweet = self.tokenizer.encode(sequence=self.sentiment[item], pair=tweet)
        tok_tweet_tokens = tok_tweet.tokens
        tok_tweet_ids = tok_tweet.ids 
        tok_tweet_offsets = tok_tweet.offsets[3:-1] # taking answer part

        targets = [0] * (len(tok_tweet_tokens) - 4) # removing tokens not in answer part

        if self.sentiment[item] == "positive" or self.sentiment[item] == "negative":
            sub_minus = 8
        else:
            sub_minus = 7

        for j, (offset1, offset2) in enumerate(tok_tweet_offsets):
            if sum(char_targets[offset1 - sub_minus: offset2 - sub_minus]) > 0:
                targets[j] = 1
        
        targets = [0] + [0] + [0] + targets + [0] # cls, sep and 2 more added here!!
        targets_start = [0] * len(targets)
        targets_end = [0] * len(targets)

        non_zero = np.nonzero(targets)[0]
        if len(non_zero) > 0:
            targets_start[non_zero[0]] = 1
            targets_end[non_zero[-1]] = 1

        # attention mask
        mask = [1] * len(tok_tweet_ids)
        token_type_ids = [0] * 3 + [1] * (len(tok_tweet_ids) - 3) # this will be all zeros
        padding_len = self.max_len - len(tok_tweet_ids) # pad everything after the tweet till end
        tok_tweet_offsets = [(0, 0)]*3 + tok_tweet_offsets + [(0, 0)]*(padding_len+1)
        # padding for each of mask, tok_tweet_ids, targets, targets_start, targets_end
        ids = tok_tweet_ids + [0] * padding_len
        mask += [0] * padding_len
        token_type_ids += [0] * padding_len
        targets += [0] * padding_len
        targets_start += [0] * padding_len
        targets_end += [0] * padding_len

        # If you have another column called sentiment, use the following preprocessing
        sentiment = [1,0,0] 
        if self.sentiment[item] == "positive":
            sentiment = [0,0,1]
        if self.sentiment[item] == "negative":
            sentiment = [0,1,0]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "targets_start": torch.tensor(targets_start, dtype=torch.long),
            "targets_end": torch.tensor(targets_end, dtype=torch.long),
            "orig_tweet": self.tweet[item],
            "sentiment": torch.tensor(sentiment, dtype=torch.long),
            "orig_sentiment": self.sentiment[item],
            "orig_selected": self.selected_text[item],
            "tweet_offsets":torch.tensor(tok_tweet_offsets, dtype=torch.long)
        }


# In[ ]:


# train = pd.read_csv(train_file)
# dset = TweetDataset(train.text.values, train.sentiment.values, train.selected_text.values)
# dset[57]


# In[ ]:


class TweetBertBaseCased(nn.Module):
    
    def __init__(self, config):
        super(TweetBertBaseCased, self).__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.num_recurrent_layers = 1
        self.units = 768

        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased', config=self.config)
        self.classifier = nn.Linear(self.units*4, self.num_labels)

    def forward(self, ids, mask, token_type_ids):

        bert_outputs = self.bert(ids,
                            attention_mask=mask,
                            token_type_ids=token_type_ids)
        
        hidden_outputs = bert_outputs[2]

        sequence_output = torch.cat(tuple([hidden_outputs[i] for i in [-1, -2, -3, -4]]), dim=-1)                
    
        logits = self.classifier(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


# In[ ]:


def decode_prediction(fin_outputs_start, fin_outputs_end, fin_tweet_offsets,
                      fin_orig_tweet, fin_orig_sentiment):
    
    threshold_start = np.argmax(fin_outputs_start, axis = 1)
    threshold_end = np.argmax(fin_outputs_end, axis = 1)
    all_strings = []
    
    for j in range(len(fin_orig_tweet)):

        original_tweet = fin_orig_tweet[j]
        sentiment_val = fin_orig_sentiment[j]
        tweet_offsets = fin_tweet_offsets[j]
        
        idx_start = threshold_start[j]
        idx_end = threshold_end[j]

        if idx_end < idx_start:
            idx_end = idx_start

        if sentiment_val == "neutral":
            sub_minus = 7
        else:
            sub_minus = 8       

        final_output  = ""
        for i in range(idx_start,idx_end+1):
            final_output += original_tweet[tweet_offsets[i][0]-sub_minus:tweet_offsets[i][1]-sub_minus]
            if (i+1) < len(tweet_offsets) and tweet_offsets[i][1] < tweet_offsets[i+1][0]:
                final_output += " "

        if sentiment_val == "neutral" or len(original_tweet.split()) < 2:
            final_output = original_tweet # just select the whole sentence
        all_strings.append(final_output)
    
    return all_strings


# In[ ]:


def train_model(model, train_data_loader, optimizer, scheduler, device):
    
    model.train()
    losses = AverageMeter()
    tk0 = tqdm(train_data_loader, total=len(train_data_loader))

    for bi, batch in enumerate(tk0):
        ids = batch["ids"]
        token_type_ids = batch["token_type_ids"]
        mask = batch["mask"]
        targets_start = batch["targets_start"]
        targets_end = batch["targets_end"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)

        optimizer.zero_grad()
        o1, o2 = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        loss = loss_fn(o1, o2, targets_start, targets_end)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward() 
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)


def eval_model(model, valid_data_loader, device):
    
    model.eval()

    fin_outputs_start = []
    fin_outputs_end = []
    fin_orig_tweet = []
    fin_orig_sentiment = []
    fin_orig_selected = []
    fin_tweet_offsets = []

    with torch.no_grad():
        for bi, batch in enumerate(valid_data_loader):
            ids = batch["ids"]
            token_type_ids = batch["token_type_ids"]
            mask = batch["mask"]
            tweet_offsets = batch["tweet_offsets"]

            # we need to calculate jaccard, so we need tweet tokens
            orig_sentiment = batch["orig_sentiment"]
            orig_selected = batch["orig_selected"]
            orig_tweet = batch["orig_tweet"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)

            o1, o2 = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            
            fin_outputs_start.append(F.softmax(o1, dim=1).cpu().detach().numpy())
            fin_outputs_end.append(F.softmax(o2, dim=1).cpu().detach().numpy())
            fin_tweet_offsets.extend(tweet_offsets.cpu().detach().numpy())

            fin_orig_sentiment.extend(orig_sentiment)
            fin_orig_selected.extend(orig_selected)
            fin_orig_tweet.extend(orig_tweet)

    # now, construct the final text we will submit to Kaggle
    fin_outputs_start = np.vstack(fin_outputs_start)
    fin_outputs_end = np.vstack(fin_outputs_end)
    
    all_strings = decode_prediction(fin_outputs_start, fin_outputs_end, fin_tweet_offsets,
                                    fin_orig_tweet, fin_orig_sentiment)
    
    jaccards = []
    for i in range(len(fin_orig_tweet)):
        target_string, final_output  = fin_orig_selected[i], all_strings[i]
        jac = jaccard(target_string, final_output)
        jaccards.append(jac)
    mean_jac = np.mean(jaccards)
    
    return mean_jac, all_strings


# In[ ]:


bert_config = BertConfig.from_pretrained('bert-base-uncased')
bert_config.num_labels = 2
bert_config.output_hidden_states = True
    
train = pd.read_csv(train_file).dropna().reset_index(drop=True)
all_scores = []
prediction_bert = ['']*len(train)
    
kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
for fold, (tr_ind, val_ind) in enumerate(kf.split(train, train['sentiment'])):
    
    if fold == 0:
        
        print(f'Fold no {fold+1}:')

        x_train = train.iloc[tr_ind].reset_index(drop=True)
        x_val = train.iloc[val_ind].reset_index(drop=True)        

        train_dataset = TweetDataset(tweet = x_train.text.values,
                                     sentiment = x_train.sentiment.values,
                                     selected_text = x_train.selected_text.values)

        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=train_batch_size,
                                                        num_workers=4)

        valid_dataset = TweetDataset(tweet = x_val.text.values,
                                     sentiment = x_val.sentiment.values,
                                     selected_text = x_val.selected_text.values)

        valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                        batch_size=valid_batch_size,
                                                        num_workers=1)

        device = torch.device("cuda")
        model = TweetBertBaseCased(config=bert_config)
        model.to(device)

        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

        num_train_steps = int(len(x_train) / train_batch_size * epochs)
        optimizer = AdamW(optimizer_parameters, lr=3e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_train_steps
        )

        model, optimizer = amp.initialize(model, optimizer, opt_level="O2", 
                                          keep_batchnorm_fp32=True, loss_scale="dynamic",
                                          verbosity = 0)
        print("Training....")

        best_jaccard = 0
        for epoch in range(epochs):
            train_model(model, train_data_loader, optimizer, scheduler, device)
            jaccard_score, out_strings = eval_model(model, valid_data_loader, device)

            print(f"Jaccard Score = {jaccard_score}")
            if jaccard_score > best_jaccard:
                torch.save(model.state_dict(), str(fold)+modle_path)
                best_jaccard = jaccard_score
                count = 0
                for i in val_ind:
                    prediction_bert[i] = out_strings[count]
                    count += 1
            break
        all_scores.append(best_jaccard)


# In[ ]:


print(all_scores)


# In[ ]:


train['predictions'] = prediction_bert
train.loc[(train.sentiment == 'negative') | (train.sentiment=='positive')][:10]

