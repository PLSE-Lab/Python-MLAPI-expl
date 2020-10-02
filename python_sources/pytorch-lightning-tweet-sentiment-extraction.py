#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install ../input/pytorchlightning-071/pytorch-lightning-0.7.1/pytorch-lightning-0.7.1')


# In[ ]:


import numpy as np
import pandas as pd
from pathlib import Path

import os
import random

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from collections import Counter
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from transformers import AlbertModel, AlbertTokenizer,RobertaForQuestionAnswering,BertPreTrainedModel,RobertaTokenizer
import tokenizers
import pytorch_lightning as pl
from transformers import RobertaTokenizer, RobertaForQuestionAnswering,RobertaConfig,RobertaModel,RobertaForMaskedLM
import torch

from tqdm import tqdm_notebook as tqdm
import itertools
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


# In[ ]:


pl.__version__


# In[ ]:


get_ipython().system('ls ../input/tweet-sentiment-extraction')


# In[ ]:


get_ipython().system('ls ../input/roberta-base')


# In[ ]:


#df_train = pd.read_csv('../input/tweet-train-folds/train_folds.csv')
df_train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
df_test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
submission = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')


# In[ ]:


df_train = df_train.dropna()
df_train.shape


# In[ ]:


def remove_multiple_dot(line):
   # print(line)
    line = re.sub('\.\.+', ' ', line) 
    #line = re.sub('\.', '', line)
    return line


# In[ ]:


def remove_single_multiple_dot(line):
    line = re.sub('\.\.+', ' ', line) 
    line = re.sub('\.', '', line)
    return line


# In[ ]:


#df_train['text'] = df_train['text'].apply(lambda x :remove_multiple_dot(x))
#df_train['selected_text'] = df_train['selected_text'].apply(lambda x :remove_single_multiple_dot(x))


# In[ ]:


#df_test['text'] = df_test['text'].apply(lambda x :remove_multiple_dot(x))


# In[ ]:


df_train, df_val = train_test_split(df_train, train_size=0.8,stratify=df_train['sentiment'])


# In[ ]:


df_train.head()


# In[ ]:


df_train.shape


# In[ ]:


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)


# In[ ]:


for col in df_train:
    if sum(df_train[col].isnull()) >  0:
        print(col,sum(df_train[col].isnull()) )


# In[ ]:


for col in df_val:
    if sum(df_val[col].isnull()) >  0:
        print(col,sum(df_val[col].isnull()) )


# In[ ]:


for col in df_test:
    if sum(df_test[col].isnull()) >  0:
        print(col,sum(df_test[col].isnull()) )


# In[ ]:


df_val.head()


# In[ ]:


df_test.head()


# In[ ]:


def process_data1(tweet, selected_text, sentiment, tokenizer, max_len):
    #print(tweet)
    #print(selected_text)
    #tweet = " " + " ".join(str(tweet).split())
    #selected_text = " " + " ".join(str(selected_text).split())
    tweet = tweet.strip()
    selected_text = selected_text.strip()
    print(tweet)
    print(selected_text)
    tweet_encoded = tokenizer.encode(tweet, add_special_tokens=True)
    print(tweet_encoded)
    selected_text_encoded = tokenizer.encode(selected_text, add_special_tokens=True)
    print(selected_text_encoded)
    sentiment_encoded = tokenizer.encode(sentiment, add_special_tokens=True)
    input_ids = tokenizer.build_inputs_with_special_tokens(sentiment_encoded,tweet_encoded)
    token_type_ids = tokenizer.create_token_type_ids_from_sequences(sentiment_encoded,tweet_encoded)
    start_id = None
    end_id = None
    for i in range(0,len(input_ids)):
        #print(selected_text_encoded[1:-1],input_ids[i:len(selected_text_encoded)+i],input_ids[i:len(selected_text_encoded)+i][1:-1])
        if selected_text_encoded[1:-1] == input_ids[i:len(selected_text_encoded)+i][1:-1]:
            start_id = i
            end_id = i + len(selected_text_encoded)
            print(start_id,end_id)
            break
        #else:
            #print(selected_text_encoded[1:-1] , input_ids[i:len(selected_text_encoded)+i][1:-1])
    mask = [1] * len(token_type_ids)
    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([1] * padding_length)
    #print('--------')
    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'target_start_id': start_id,
        'target_end_id': end_id,
        'tweet': tweet,
        'selected_text': selected_text,
        'sentiment': sentiment
    }


# In[ ]:


def process_data(tweet, selected_text, sentiment, tokenizer, max_len):
    tweet = " " + " ".join(str(tweet).split())
    selected_text = " " + " ".join(str(selected_text).split())
    #print(tweet)
    #print(selected_text)
    len_st = len(selected_text) - 1
    idx0 = None
    idx1 = None

    for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
        if " " + tweet[ind: ind+len_st] == selected_text:
            idx0 = ind
            idx1 = ind + len_st - 1
            break

    char_targets = [0] * len(tweet)
    if idx0 != None and idx1 != None:
        for ct in range(idx0, idx1 + 1):
            char_targets[ct] = 1
    
    tok_tweet = tokenizer.encode(tweet)
    input_ids_orig = tok_tweet.ids
    tweet_offsets = tok_tweet.offsets
    
    target_idx = []
    for j, (offset1, offset2) in enumerate(tweet_offsets):
        if sum(char_targets[offset1: offset2]) > 0:
            target_idx.append(j)
    
    targets_start = target_idx[0]
    targets_end = target_idx[-1]

    sentiment_id = {
        'positive': 1313,
        'negative': 2430,
        'neutral': 7974
    }
    
    input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
    token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
    targets_start += 4
    targets_end += 4
    #print(targets_start,targets_end)
    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([1] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'target_start_id': targets_start,
        'target_end_id': targets_end,
        'tweet': tweet,
        'selected_text': selected_text,
        'sentiment': sentiment,
        'offsets': tweet_offsets
    }


# In[ ]:


def process_data_test1(tweet, selected_text, sentiment, tokenizer, max_len):
    tweet_encoded = tokenizer.encode(tweet, add_special_tokens=True)
    sentiment_encoded = tokenizer.encode(sentiment, add_special_tokens=True)
    input_ids = tokenizer.build_inputs_with_special_tokens(sentiment_encoded,tweet_encoded)
    token_type_ids = tokenizer.create_token_type_ids_from_sequences(sentiment_encoded,tweet_encoded)
    mask = [1] * len(token_type_ids)
    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([1] * padding_length)
    
    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'tweet': tweet,
        'sentiment': sentiment
    }


# In[ ]:


def process_data_test(tweet, sentiment, tokenizer, max_len):
    tweet = " " + " ".join(str(tweet).split())
    #len_st = len(selected_text) - 1
    tok_tweet = tokenizer.encode(tweet)
    input_ids_orig = tok_tweet.ids
    tweet_offsets = tok_tweet.offsets

    sentiment_id = {
        'positive': 1313,
        'negative': 2430,
        'neutral': 7974
    }
    
    input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
    token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([1] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
    
    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'tweet': tweet,
        'sentiment': sentiment,
        'offsets': tweet_offsets
    }


#         input_ids,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         start_positions=None,
#         end_positions=None,

# In[ ]:


class Training_Dataset(Dataset):
    def __init__(self,df):
        super().__init__()
        self.tweet = df['text']
        self.sentiment = df['sentiment']
        self.selected_text = df['selected_text']
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN
        self.text_id = df['textID'].values

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        data = process_data(
            self.tweet[item], 
            self.selected_text[item], 
            self.sentiment[item],
            self.tokenizer,
            self.max_len
        )
        return {
#            'idx': torch.tensor(item, dtype=torch.long),
            'idx':item,
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'target_start_id': torch.tensor(data["target_start_id"], dtype=torch.long),
            'target_end_id': torch.tensor(data["target_end_id"], dtype=torch.long),
            'tweet': data["tweet"],
            'selected_text': data["selected_text"],
            'sentiment': data["sentiment"],
            'offsets': torch.tensor(data["offsets"], dtype=torch.long)
        }


# In[ ]:


#textID	text	selected_text	sentiment	kfold


# In[ ]:


class Test_Dataset(Dataset):
    def __init__(self,df):
        super().__init__()
        self.tweet = df['text'].values
        self.sentiment = df['sentiment'].values
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN
        self.text_id = df['textID'].values

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        data = process_data_test(
            self.tweet[item], 
            self.sentiment[item],
            self.tokenizer,
            self.max_len
        )
        
        return {
         #   'idx': torch.tensor(item, dtype=torch.long),
            'idx':self.text_id[item],
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'tweet': data["tweet"],
            'sentiment': data["sentiment"],
            'offsets': torch.tensor(data["offsets"], dtype=torch.long)
        }


# In[ ]:


class TweetBaseSuperModule(pl.LightningModule):
    def __init__(self, model, tokenizer, prediction_save_path):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prediction_save_path = prediction_save_path
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(1024 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
        self.loss_average_val =0
        self.loss_average_train = 0

    def get_device(self):
        return self.bertmodel.state_dict()['bert.embeddings.word_embeddings.weight'].device

    def save_predictions(self, idx,start_positions, end_positions,filtered_output):
        d = pd.DataFrame({'text_ID':idx,'start_position':start_positions, 'end_position':end_positions,'selected_text':filtered_output})
        d.to_csv(self.prediction_save_path, index=False)
        
    def save_predictions1(self, idx,start_positions, end_positions):
        d = pd.DataFrame({'text_ID':idx,'start_position':start_positions, 'end_position':end_positions})
        d.to_csv(self.prediction_save_path, index=False)
        
    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.model(
            ids,
            mask,
            token_type_ids=token_type_ids
        ) 
        out = torch.cat((out[-1], out[-2]), dim=-1) 
        out = self.drop_out(out) 
        logits = self.l0(out) 
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1) 
        end_logits = end_logits.squeeze(-1) 
        return start_logits, end_logits
    
    
    def loss(self,start_logits, end_logits, start_positions, end_positions):
        """
        Return the sum of the cross entropy losses for both the start and end logits
        """
        loss_fct = nn.CrossEntropyLoss()
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss)
        return total_loss
    
    def extract_selected_text(self,original_tweet, sentiment_val,idx_start, idx_end,offsets):
        #print(original_tweet)
        #print(sentiment_val)
        #print(idx_start)
        if idx_end < idx_start:
            idx_end = idx_start
        filtered_output  = ""
        for ix in range(idx_start, idx_end + 1):
            filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
            if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
                filtered_output += " "
        if sentiment_val == "neutral" or len(original_tweet.split()) < 2:
            filtered_output = original_tweet
        return filtered_output
    
    def extract_selected_text_batch(self,tweet, sentiment,idx_start, idx_end,offsets):
        filtered_output_ls = []
        for tweet_value,sentiment_value,start_value,end_value,offset_value in zip(tweet,sentiment,idx_start,idx_end,offsets):
            filtered_output = self.extract_selected_text(
                original_tweet=tweet_value,
                sentiment_val=sentiment_value,
                idx_start=start_value,
                idx_end=end_value,
                offsets=offset_value
            )
            filtered_output_ls.append(filtered_output)
        return filtered_output_ls
            

    def training_step(self, batch, batch_nb):
        """
        (batch) -> (dict or OrderedDict)
        # Caution: key for loss function must exactly be 'loss'.
        """
        idx = batch['idx']
        #ids, mask, token_type_ids
        start_logits, end_logits = self.forward(batch['ids'],batch['mask'],batch['token_type_ids'])
        loss = self.loss(start_logits, end_logits,batch['target_start_id'],batch['target_end_id'])
        self.loss_average_train = (self.loss_average_train + loss)/(batch_nb+1)
        if batch_nb % 200 == 0: 
            print(f"TRAIN : Batch {batch_nb} Average Loss {self.loss_average_train} batch loss: {loss}")
        return {'loss':loss, 'idx':idx,'total_average_train_loss':self.loss_average_train}

    def validation_step(self, batch, batch_nb):
        """
        (batch) -> (dict or OrderedDict)
        # Caution: key for loss function must exactly be 'loss'.
        """
        idx = batch['idx']
        start_logits, end_logits = self.forward(batch['ids'],batch['mask'],batch['token_type_ids'])
        loss = self.loss(start_logits, end_logits,batch['target_start_id'],batch['target_end_id'])
        self.loss_average_val = (self.loss_average_val + loss)/(batch_nb+1)
        if batch_nb % 100 == 0: 
            print(f"VAL : Batch {batch_nb} Average Loss {self.loss_average_val} batch loss: {loss}")
        return {'loss':loss, 'idx':idx,'total_average_val_loss':self.loss_average_val}

    def test_step(self, batch, batch_nb):
        """
        (batch) -> (dict or OrderedDict)
        """
        idx = batch['idx']
        start_scores = self.forward(batch['ids'],batch['mask'],batch['token_type_ids'])[0]
        end_scores = self.forward(batch['ids'],batch['mask'],batch['token_type_ids'])[1]
        tweet = batch['tweet']
        offsets = batch['offsets']
        sentiment = batch['sentiment']
        return {'start_scores':start_scores, 'end_scores':end_scores, 'idx':idx,'tweet':tweet,'offsets':offsets,'sentiment':sentiment}

    def training_end(self, outputs):
        """
        outputs(dict) -> loss(dict or OrderedDict)
        # Caution: key must exactly be 'loss'.
        """
        #train_num_steps = int(len(df_train) / TRAIN_BATCH_SIZE * EPOCHS)
        #l = outputs['loss']
        #tl = outputs['total_train_loss']/train_num_steps
        #print(f"TRAIN STEP END : Total Return Loss {l}, TOTAL LOSS {tl}, TRAIN STEPS : {train_num_steps}")
        return {'loss':outputs['loss']}

    def validation_end(self, outputs):
        """
        For single dataloader:
            outputs(list of dict) -> (dict or OrderedDict)
        For multiple dataloaders:
            outputs(list of (list of dict)) -> (dict or OrderedDict)
        """        
        return {'loss':torch.mean(torch.tensor([output['loss'] for output in outputs])).detach()}

    def test_end(self, outputs):
        """
        For single dataloader:
            outputs(list of dict) -> (dict or OrderedDict)
        For multiple dataloaders:
            outputs(list of (list of dict)) -> (dict or OrderedDict)
        """
        start_scores = torch.cat([output['start_scores'] for output in outputs]).detach().cpu().numpy()
        start_positions = np.argmax(start_scores, axis=1) - 1

        end_scores = torch.cat([output['end_scores'] for output in outputs]).detach().cpu().numpy()
        end_positions = np.argmax(end_scores, axis=1) - 1
        idx = [output['idx'] for output in outputs]
        idx = list(itertools.chain.from_iterable(idx))
        
        tweet = [output['tweet'] for output in outputs]
        tweet = list(itertools.chain.from_iterable(tweet))

        offsets = [output['offsets'] for output in outputs]
        offsets = list(itertools.chain.from_iterable(offsets))
        
        sentiment = [output['sentiment'] for output in outputs]
        sentiment = list(itertools.chain.from_iterable(sentiment))
        
        filtered_output= self.extract_selected_text_batch(tweet, sentiment,start_positions, end_positions,offsets)
        self.save_predictions(idx,start_positions, end_positions,filtered_output)
        return {}

    #def configure_optimizers(self):
    #    return optim.Adam(self.parameters(), lr=2e-5)
    
    def configure_optimizers(self):
        num_train_steps = int(len(df_train) / TRAIN_BATCH_SIZE * EPOCHS)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        
        optimizer = AdamW(optimizer_parameters, lr=3e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=num_train_steps)
        return [optimizer], [scheduler]


    @pl.data_loader
    def train_dataloader(self):
        pass

    @pl.data_loader
    def val_dataloader(self):
        pass

    @pl.data_loader
    def test_dataloader(self):
        pass


# In[ ]:


np.array([1,2,4]).shape


# In[ ]:


class Tweet_Model(TweetBaseSuperModule):
    def __init__(self, bertmodel, tokenizer, prediction_save_path):
        super().__init__(bertmodel, tokenizer, prediction_save_path)

    @pl.data_loader
    def train_dataloader(self):
        return train_dl

    @pl.data_loader
    def val_dataloader(self):
        return val_dl

    @pl.data_loader
    def test_dataloader(self):
        return test_dl


# In[ ]:


ls ../input/roberta-transformers-pytorch/roberta-large


# In[ ]:


MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 2
DEBUG_MODE = False

ROBERTA_PATH = "../input/roberta-transformers-pytorch/roberta-large/"
TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab_file=f"{ROBERTA_PATH}vocab.json", 
    merges_file=f"{ROBERTA_PATH}merges.txt", 
    lowercase=True,
    add_prefix_space=True
)

#tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model_config = RobertaConfig.from_pretrained(f"{ROBERTA_PATH}config.json")
model_config.output_hidden_states = True
model = RobertaModel.from_pretrained(f"{ROBERTA_PATH}pytorch_model.bin", config=model_config)


# In[ ]:


#tweet, sentiment, selected_text,tokenizer,max_len
train_ds = Training_Dataset(df_train)
val_ds = Training_Dataset(df_val)
test_ds = Test_Dataset(df_test)


# In[ ]:


train_dl = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, num_workers=4)
val_dl = DataLoader(val_ds, batch_size=VALID_BATCH_SIZE, num_workers=4)
test_dl = DataLoader(test_ds, batch_size=VALID_BATCH_SIZE, num_workers=4)


# In[ ]:


model = Tweet_Model(model, TOKENIZER, 'pred.csv')
model


# In[ ]:


1024*2


# In[ ]:


trainer = pl.Trainer(gpus=1,max_nb_epochs=EPOCHS, fast_dev_run=DEBUG_MODE)


# In[ ]:


trainer.fit(model)


# In[ ]:


trainer.test()


# In[ ]:


pred = pd.read_csv('pred.csv')


# In[ ]:


pred = pred[['text_ID','selected_text']]


# In[ ]:


pred.columns=['textID','selected_text']


# In[ ]:


def fillna_by_text(val):
    if val['selected_text']=='CODECXXX001':
        return ' '.join(val['text'].split(' ')[0:3])
    else:
        return val['selected_text']


# In[ ]:


submission_final = pred.merge(df_test,on='textID',how='inner')
submission_final = submission_final.fillna('CODECXXX001')
submission_final['selected_text1']=submission_final.apply(lambda x : fillna_by_text(x),axis=1)
submission_final = submission_final[['textID','selected_text1']]
submission_final.columns=['textID','selected_text']
submission_final.to_csv('submission.csv',index=False)


# In[ ]:


#pred.to_csv('submission.csv',index=False)


# In[ ]:


submission_final.shape,submission.shape


# In[ ]:


submission_final

