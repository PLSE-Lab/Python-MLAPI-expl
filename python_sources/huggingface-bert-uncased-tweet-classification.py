#!/usr/bin/env python
# coding: utf-8

# # Huggingface Transformer Pretrained Model
# 
# This is very basic introductory repository which uses pretrained huggingface tokenizer and bert model.<br>
# This uses on 'text' field.<br>
# <br>
# Repository can be found: https://github.com/nachiket273/Tweet_Classification_Huggingface_Wandb <br>
# <br>
# 
# Following steps are performed: <br>
# <br>
# 1) Pre-processing:<br>
#     i) Remove urls , non-ascii characters, emoji's,  punctuations, contractions (standard preprocessing <br>
#     most of the notebooks :) <br>
#     ii) to-lower <br>
# <br>   
# 2) Class - Balancing : Sampling from class with more training example to balance class.<br>
# <br>
# 3) Model, Optimizer and Other configurations:<br>
# tokenizer and bert model = 'bert-base-uncased' <br>
# optimizer= AdamW <br>
# scheduler linear with warmup <br>
# start_lr = 2e-5 <br>
# train_bs = 8 <br>
# valid_bs = 8 <br>
# epochs = 5 <br>
# max_len = 160 <br>
# dropout_ratio = 0.1 <br>
# warmup_epochs = 0 <br>
# test_size = 0.2 <br>
# <br>
# 4) Ran for 3 different seed - 42, 11, 2020 and averaged the predictions.<br>
# <br>
# 5)Repository also contains basic visualizations and tracking using 'weights and biases'.<br>
# <br>

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


import numpy as np 
import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import time
import torch
import transformers
from transformers import BertConfig
from transformers import get_linear_schedule_with_warmup


# In[ ]:


import sys
sys.path.insert(1, '../input/tweet-classification-huggingface-wandb1/')


# In[ ]:


import config
import dataset
import model
import train
import util


# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[ ]:


warmup_epochs = 0
epochs = 5
model_name = 'bert-base-uncased'
test_size = 0.3
dropout = 0.1
num_classes = 2
linear_in = 768
max_len = 160
train_bs = 16
valid_bs = 8
start_lr = 2e-5

SEED = 42


# In[ ]:


util.set_seed(SEED)


# In[ ]:


train_df = pd.read_csv('../input/nlp-getting-started/train.csv')
test_df = pd.read_csv('../input/nlp-getting-started/test.csv')


# # **Preprocessing**

# In[ ]:


import preprocess
import re
import string


# In[ ]:


ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]
train_df = preprocess.fix_erraneous(train_df, ids_with_target_error, 0)


# In[ ]:


def remove_tabs_newlines(text):
    text = re.sub(r'\t', ' ', text) # remove tabs
    text = re.sub(r'\n', ' ', text) # remove line jump
    return text


# In[ ]:


train_df['text'] = train_df['text'].apply(lambda k : remove_tabs_newlines(k))
test_df['text'] = test_df['text'].apply(lambda k : remove_tabs_newlines(k))


# In[ ]:


def remove_url(text):
    return re.sub(r"http\S+", "URL", text)


# In[ ]:


def remove_non_ascii(text):
    return ''.join([x for x in text if x in string.printable])


# In[ ]:


def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# In[ ]:


def remove_usrname(text):
    text = re.sub(r'@\S{0,}', ' USER ', text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'\b(USER)( \1\b)+', r'\1', text)
    return text


# In[ ]:


# A list of contractions from 
# http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "thx"   : "thanks"
}

def remove_contractions(text):
    return contractions[text.lower()] if text.lower() in contractions.keys() else text


# In[ ]:


def remove_punctuations(text):
    return text.translate(str.maketrans('', '', string.punctuation))


# In[ ]:


train_df['text'] = train_df['text'].apply(lambda k: remove_usrname(k))
train_df['text'] = train_df['text'].apply(lambda k: remove_contractions(k))
train_df['text'] = train_df['text'].apply(lambda k: remove_url(k))
train_df['text'] = train_df['text'].apply(lambda k: remove_emoji(k))
train_df['text'] = train_df['text'].apply(lambda k: remove_non_ascii(k))
train_df['text'] = train_df['text'].apply(lambda k: remove_punctuations(k))


# In[ ]:


test_df['text'] = test_df['text'].apply(lambda k: remove_usrname(k))
test_df['text'] = test_df['text'].apply(lambda k: remove_contractions(k))
test_df['text'] = test_df['text'].apply(lambda k: remove_url(k))
test_df['text'] = test_df['text'].apply(lambda k: remove_emoji(k))
test_df['text'] = test_df['text'].apply(lambda k: remove_non_ascii(k))
test_df['text'] = test_df['text'].apply(lambda k: remove_punctuations(k))


# In[ ]:


def cleanup(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


# In[ ]:


def replace_nums(text):
    text = re.sub(r'^\d\S{0,}| \d\S{0,}| \d\S{0,}$', ' NUMBER ', text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'\b(NUMBER)( \1\b)+', r'\1', text)
    text = re.sub(r"[0-9]", " ", text)
    return text


# In[ ]:


train_df['text'] = train_df['text'].apply(lambda k: replace_nums(k))
test_df['text'] = test_df['text'].apply(lambda k: replace_nums(k))


# In[ ]:


train_df['text'] = train_df['text'].apply(lambda k: cleanup(k))
test_df['text'] = test_df['text'].apply(lambda k: cleanup(k))


# In[ ]:


train_df['text'][:5]


# # Split Dataset

# In[ ]:


train_y = train_df['target']
train_df.drop(['target'], axis=1, inplace=True)


# In[ ]:


X_train, X_test, y_train, y_test         = train_test_split(train_df['text'], train_y, random_state=SEED, test_size=test_size, stratify=train_y.values) 


# # **Encode and get dataloaders**

# In[ ]:


tokenizer = transformers.BertTokenizer.from_pretrained(
        model_name,
        do_lower_case=True
    )


# In[ ]:


train_dataset = dataset.BertDataset(
        text=X_train.values,
        tokenizer= tokenizer,
        max_len= max_len,
        target=y_train.values
    )


# In[ ]:


valid_dataset = dataset.BertDataset(
        text=X_test.values,
        tokenizer= tokenizer,
        max_len= max_len,
        target=y_test.values
    )


# In[ ]:


train_dl = torch.utils.data.DataLoader(
        train_dataset,
        train_bs,
        shuffle=True,
        num_workers=4
    )


# In[ ]:


valid_dl = torch.utils.data.DataLoader(
        valid_dataset,
        valid_bs,
        shuffle=True,
        num_workers=1
    )


# # Model

# In[ ]:


bert = model.BertHf(model_name, dp=dropout, num_classes=num_classes, linear_in=linear_in)
bert = bert.to(device)


# In[ ]:


for param in bert.parameters():
    param.requires_grad = True


# In[ ]:


no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
params = list(bert.named_parameters())
modified_params = [
    {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.1},
    {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]


# In[ ]:


optim = torch.optim.AdamW(modified_params, lr=start_lr, eps=1e-8)


# In[ ]:


total_steps = int(len(train_df) * epochs / train_bs)
warmup_steps = int(len(train_df) * warmup_epochs / train_bs)


# In[ ]:


sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)


# In[ ]:


criterion = torch.nn.CrossEntropyLoss()


# # Now Train

# In[ ]:


train_stat = util.AvgStats()
valid_stat = util.AvgStats()


# In[ ]:


best_acc = 0.0
best_model_file = str(model_name) + '_best.pth.tar'


# In[ ]:


print("\nEpoch\tTrain Acc\tTrain Loss\tTrain F1\tValid Acc\tValid Loss\tValid F1")
for i in range(epochs):
    start = time.time()
    losses, ops, targs = train.train(train_dl, bert, criterion, optim, sched, device)
    duration = time.time() - start
    train_acc = accuracy_score(targs, ops)
    train_f1_score = f1_score(targs, ops)
    train_loss = sum(losses)/len(losses)
    train_prec = precision_score(targs, ops)
    train_rec = recall_score(targs, ops)
    train_roc_auc = roc_auc_score(targs, ops)
    train_stat.append(train_loss, train_acc, train_f1_score, train_prec, train_rec, train_roc_auc, duration)
    start = time.time()
    lossesv, opsv, targsv = train.test(valid_dl, bert, criterion, device)
    duration = time.time() - start
    valid_acc = accuracy_score(targsv, opsv)
    valid_f1_score = f1_score(targsv, opsv)
    valid_loss = sum(lossesv)/len(lossesv)
    valid_prec = precision_score(targsv, opsv)
    valid_rec = recall_score(targsv, opsv)
    valid_roc_auc = roc_auc_score(targsv, opsv)
    valid_stat.append(valid_loss, valid_acc, valid_f1_score, valid_prec, valid_rec, valid_roc_auc, duration)

    if valid_acc > best_acc:
        best_acc = valid_acc
        util.save_checkpoint(bert, True, best_model_file)
        tfpr, ttpr, _ = roc_curve(targs, ops)
        train_stat.update_best(ttpr, tfpr, train_acc, i)
        vfpr, vtpr, _ = roc_curve(targsv, opsv)
        valid_stat.update_best(vtpr, vfpr, best_acc, i)

    print("\n{}\t{:06.8f}\t{:06.8f}\t{:06.8f}\t{:06.8f}\t{:06.8f}\t{:06.8f}".format(i+1, train_acc*100, train_loss, 
                                                train_f1_score, valid_acc*100, 
                                                valid_loss, valid_f1_score))


# In[ ]:


print("Summary of best run::")
print("Best Accuracy: {}".format(valid_stat.best_acc))
print("Roc Auc score: {}".format(valid_stat.roc_aucs[valid_stat.best_epoch]))
print("Loss: {}".format(valid_stat.losses[valid_stat.best_epoch]))
print("Area Under Curve: {}".format(auc(valid_stat.fprs, valid_stat.tprs)))


# # Now make predictions

# In[ ]:


test_dataset = dataset.BertDataset(
        text=test_df.text.values,
        tokenizer= tokenizer,
        max_len= max_len
    )


# In[ ]:


test_dl = torch.utils.data.DataLoader(
        test_dataset,
        1,
        shuffle=False,
        num_workers=1
    )


# In[ ]:


util.load_checkpoint(bert, best_model_file)


# In[ ]:


_, opst, _ = train.test(test_dl, bert, criterion, device)


# # Submission

# In[ ]:


sub_csv = pd.DataFrame(columns=['id', 'target'])


# In[ ]:


sub_csv['id'] = test_df['id']
sub_csv['target'] = opst


# In[ ]:


sub_csv.to_csv('submission.csv', index=False)

