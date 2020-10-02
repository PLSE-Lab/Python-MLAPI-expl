#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# https://github.com/VinAIResearch/BERTweet

get_ipython().system('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
get_ipython().system('python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev')

get_ipython().system('export XLA_USE_BF16=1')

get_ipython().system('pip install transformers')
get_ipython().system('pip install fairseq')
get_ipython().system('pip install fastBPE')


# In[ ]:


# BERTWEET Model
get_ipython().system('wget https://public.vinai.io/BERTweet_base_transformers.tar.gz')
get_ipython().system('tar -xzf *.tar.gz')


# In[ ]:


import os
import torch
import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from sklearn import model_selection
from sklearn import metrics
import transformers
import tokenizers

import argparse
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


LEARNING_RATE = 4e-5
MAX_LEN = 156
TRAIN_BATCH_SIZE = 52
VALID_BATCH_SIZE = 32
EPOCHS = 3
INPUT_PATH = 
OUTPUT_PATH = 
TRAINING_FILE = f"{INPUT_PATH}tweet-sentiment-extraction/FILE.csv"

#Load BPE encoder 
key = argparse.Namespace(bpe_codes= f"bpe.codes")
bpe = fastBPE(key)

# Load the dictionary  
vocab = Dictionary()
vocab.add_from_file("dict.txt")


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


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# In[ ]:


class TweetDataset:
    def __init__(self, tweets, sentiments, selected_texts):
        self.tweets = [' '+' '.join(str(tweet).split()) for tweet in tweets]
        self.sentiments = [' '+' '.join(str(sentiment).split()) for sentiment in sentiments]
        self.selected_texts = [' '+' '.join(str(selected_text).split()) for selected_text in selected_texts]
        self.max_len = MAX_LEN
        
    
    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        e_tweet = '<s> ' + bpe.encode(self.tweets[item]) + ' </s>'
        enc_tweet = vocab.encode_line(e_tweet, append_eos=False, add_if_not_exist=False).long().tolist()

        e_sentiment = '</s> ' + bpe.encode(self.sentiments[item]) + ' </s>'
        enc_sentiment = vocab.encode_line(e_sentiment, append_eos=False, add_if_not_exist=False).long().tolist()

        enc_tweet_sentiment = enc_tweet + enc_sentiment

        padding_len = self.max_len - len(enc_tweet_sentiment)
        input_ids = enc_tweet_sentiment + ([0] * padding_len)
        attention_mask = ([1] * len(enc_tweet_sentiment)) + ([0] * padding_len)

        start_index, end_index = 0, 0
        token_type_ids = [0] * self.max_len

        e_selected_text_ids = bpe.encode(self.selected_texts[item])
        enc_selected_text_ids = vocab.encode_line(e_selected_text_ids, append_eos=False, add_if_not_exist=False).long().tolist()

        for j in (i for i,e in enumerate(enc_tweet_sentiment) if e == enc_selected_text_ids[0]):
            if enc_tweet_sentiment[j:j+len(enc_selected_text_ids)] == enc_selected_text_ids:
                start_index = j
                end_index = j+(len(enc_selected_text_ids))

        return {
            'ids': torch.tensor(input_ids, dtype=torch.long),
            'mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets_start': torch.tensor(start_index, dtype=torch.long),
            'targets_end': torch.tensor(end_index, dtype=torch.long),
            'orig_tweet': self.tweets[item],
            'orig_selected': self.selected_texts[item],
            'sentiment': self.sentiments[item]
        }


# In[ ]:


class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(
            "BERTweet_base_transformers/model.bin",
            config=conf
        )
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
    
    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.drop_out(out)
        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


# In[ ]:



def loss_fn(start_logits, end_logits, start_positions, end_positions):
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss)
    return total_loss

def train_fn(data_loader, model, optimizer, device, num_batches, scheduler=None):
    model.train()
    tk0 = tqdm(data_loader, total=len(data_loader), desc="Training")
    for bi, d in enumerate(tk0):
        ids = d["ids"].cuda()
        token_type_ids = d["token_type_ids"].cuda()
        mask = d["mask"].cuda()
        targets_start = d["targets_start"].cuda()
        targets_end = d["targets_end"].cuda()
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_tweet = d["orig_tweet"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]

        model.zero_grad()
        outputs_start, outputs_end = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
        loss.backward()
        xm.optimizer_step(optimizer, barrier=True)
        scheduler.step()
        tk0.set_postfix(loss=loss.item())

def eval_fn(data_loader, model, device):
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()
    
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader), desc="Validating")
        for bi, d in enumerate(tk0):
            ids = d["ids"].cuda()
            token_type_ids = d["token_type_ids"].cuda()
            mask = d["mask"].cuda()
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            targets_start = d["targets_start"].cuda()
            targets_end = d["targets_end"].cuda()

            outputs_start, outputs_end = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()

            jaccard_scores = []
            for id, tweet in enumerate(orig_tweet):
                a = np.argmax(outputs_start[id])
                b = np.argmax(outputs_end[id])
                if a > b:
                    selected_text = tweet
                else:
                    n_tweet = '<s> ' + bpe.encode(tweet) + ' </s>'
                    nn_tweet = vocab.encode_line(n_tweet, append_eos=False, add_if_not_exist=False).long().tolist()
                    select = nn_tweet[a:b]
                    selected_text = bpe.decode(" ".join([vocab[i] for i in select]))
                    selected_text = selected_text.replace("<s>","").replace("</s>","")
                jaccard_scores.append(jaccard(selected_text, orig_selected[id]))

            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=loss.item())

    return jaccards.avg


# In[ ]:


dfx = pd.read_csv(TRAINING_FILE)

def run(fold):
    model_config = transformers.RobertaConfig.from_pretrained("config.json")
    model_config.output_hidden_states = True
    MX = TweetModel(conf=model_config)

    device = xm.xla_device(fold+1)
    model = MX.to(device)
    
    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

    train_dataset = TweetDataset(
        tweets=df_train.text.values,
        sentiments=df_train.sentiment.values,
        selected_texts=df_train.selected_text.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=1
    )

    valid_dataset = TweetDataset(
        tweets=df_valid.text.values,
        sentiments=df_valid.sentiment.values,
        selected_texts=df_valid.selected_text.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        num_workers=1
    )

    num_batches = int(len(df_train)/TRAIN_BATCH_SIZE)
    num_train_steps = num_batches * EPOCHS

    optimizer = transformers.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE
    )
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    best_jac = 0
    
    for epoch in range(EPOCHS):
        train_fn(
            train_data_loader, 
            model, 
            optimizer, 
            device,
            num_batches,
            scheduler
        )

        jac = eval_fn(
            valid_data_loader, 
            model, 
            device
        )
        print(f'Epoch={epoch}, Fold={fold}, Jaccard={jac}')
        if jac > best_jac:
            xm.save(model.state_dict(), f"{OUTPUT_PATH}model_{fold}.bin")
            best_jac = jac


# In[ ]:


Parallel(n_jobs=8, backend="threading")(delayed(run)(i) for i in range(8))

