#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
#from apex import amp
import random
import re
import json
from transformers import ( 
    BertTokenizer, 
    AdamW, 
    BertModel, 
    BertForPreTraining,
    BertConfig,
    get_linear_schedule_with_warmup,
    BertTokenizerFast,
    RobertaModel,
    RobertaTokenizerFast,
    RobertaConfig
)
import transformers


def to_list(tensor):
    return tensor.detach().cpu().tolist()


# In[ ]:


def process_with_offsets(args, tweet, sentiment, tokenizer):
    
    encoded = tokenizer.encode_plus(
                    sentiment,
                    tweet,
                    max_length=args.max_seq_len,
                    pad_to_max_length=True,
                    return_token_type_ids=True,
                    #return_offsets_mapping=True
                )
    
    
    encoded["tweet"] = tweet
    encoded["sentiment"] = sentiment

    return encoded


# In[ ]:


class TweetDataset:
    def __init__(self, args, tokenizer, df, mode="test"):
        
        self.mode = mode

        
        self.tweet = df.text.values
        self.sentiment = df.sentiment.values

        
        self.tokenizer = tokenizer
        self.args = args
    
    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):

        tweet = str(self.tweet[item])
        sentiment = str(self.sentiment[item])
        
        features = process_with_offsets(
                        args=self.args, 
                        tweet=tweet, 
                        sentiment=sentiment, 
                        tokenizer=self.tokenizer
                    )
        
        return {
            "input_ids":torch.tensor(features["input_ids"], dtype=torch.long),
            "token_type_ids":torch.tensor(features["token_type_ids"], dtype=torch.long),
            "attention_mask":torch.tensor(features["attention_mask"], dtype=torch.long),
            #"offsets":features["offset_mapping"],

            "tweet":features["tweet"],
            "sentiment":features["sentiment"]

        }


# In[ ]:



class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, model_path, conf):
        super(TweetModel, self).__init__(conf)
        self.xlmroberta = transformers.XLMRobertaModel.from_pretrained(model_path, config=conf)
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768 * 2, 2) #768
        torch.nn.init.normal_(self.l0.weight, std=0.02)


    def forward(self,input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        _,_, out = self.xlmroberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2]), dim=-1)
        #print(out.shape)
        out = self.drop_out(out)
        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


# In[ ]:


def test(args, test_loader, model):
    
    start_positions = []
    end_positions = []

    model.eval()

    with torch.no_grad():
        t = tqdm(test_loader)
        for step, d in enumerate(t):
            
            input_ids = d["input_ids"].to(args.device)
            attention_mask = d["attention_mask"].to(args.device)
            token_type_ids = d["token_type_ids"].to(args.device)

            logits1, logits2 = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids, 
                position_ids=None, 
                head_mask=None
            )

            logits1 = F.softmax(logits1, dim=1).cpu().data.numpy()
            logits2 = F.softmax(logits2, dim=1).cpu().data.numpy()
            
            start_positions.append(logits1)
            end_positions.append(logits2)
    
    return start_positions, end_positions


# In[ ]:


class args:
    max_seq_len = 128+64
    batch_size = 16

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.no_cuda = False if torch.cuda.is_available() else True
args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()


# In[ ]:


model_path = "../input/xlm-roberta-base/"

config = transformers.XLMRobertaConfig.from_pretrained(model_path)
config.output_hidden_states = True
tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(model_path, do_lower_case=True)
model = TweetModel(model_path, config)

model.to(args.device)

test_df = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")


# In[ ]:


test_dataset = TweetDataset(
    args=args,
    df=test_df,
    mode="test",
    tokenizer=tokenizer
)
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False
)


# In[ ]:


model_weights_list = [
    #"../input/xml-roberta-base-weights/fold_0",
    #"../input/xml-roberta-base-weights/fold_1",
    #"../input/xml-roberta-base-weights/fold_2",
    #"../input/xml-roberta-base-weights/fold_3",
    #"../input/xml-roberta-base-weights/fold_4",
    
    "../input/tweet-xlm-roberta-3fold-tpu/xlm-roberta-base/xlm-roberta-base/fold_0",
    "../input/tweet-xlm-roberta-3fold-tpu/xlm-roberta-base/xlm-roberta-base/fold_1",
    "../input/tweet-xlm-roberta-3fold-tpu/xlm-roberta-base/xlm-roberta-base/fold_2"
]

start_list = []
end_list = []

for path in model_weights_list:
    model.load_state_dict(torch.load(path, map_location=args.device))
    
    start_positions, end_positions = test(args, test_loader, model)
    
    start_positions = np.concatenate(start_positions)
    end_positions = np.concatenate(end_positions)
    
    start_list.append(start_positions)
    end_list.append(end_positions)


# In[ ]:


start_list = np.array(start_list)
end_list = np.array(end_list)


# In[ ]:


start_positions = np.argmax(np.mean(start_list, axis=0), axis=1)
end_positions = np.argmax(np.mean(end_list, axis=0), axis=1)


# In[ ]:


sub_df = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")

for i, (text, sentiment) in enumerate(zip(test_df.text.values, test_df.sentiment.values)):
    
    idx_start = start_positions[i]
    idx_end = end_positions[i]
    
    encoded = tokenizer.encode_plus(
                    sentiment,
                    text,
                    max_length=args.max_seq_len,
                    pad_to_max_length=True,
                    return_token_type_is=True,
                    #return_offsets_mapping=True
                )
    input_id = encoded["input_ids"]
    #offsets = encoded["offset_mapping"]
    
    if idx_end < idx_start:
        idx_end = idx_start
    
    filtered_output = tokenizer.decode(input_id[idx_start:idx_end+1], skip_special_tokens=True)
    
    
    #filtered_output = ""
    #for ix in range(idx_start, idx_end):
    #    filtered_output += text[offsets[ix][0]: offsets[ix][1]]
    #    if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
    #        filtered_output += " "
    
    
    if sentiment == "neutral" or len(text.split()) < 2:
        filtered_output = text
    
    filtered_output = filtered_output.strip()
    
    sub_df.loc[i, "selected_text"] = filtered_output


# In[ ]:


sub_df.to_csv("submission.csv", index=False)


# In[ ]:


sub_df


# In[ ]:




