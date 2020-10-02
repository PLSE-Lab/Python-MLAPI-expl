#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import tokenizers
import string
import torch
import transformers
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import pickle
import re
import string


# ### datasets info
# 
# roberta-pp -> preprocess + postprocess + (valideated on after preprocess) CV: 0.717+

# In[ ]:


MAX_LEN = 168
VALID_BATCH_SIZE = 8
EPOCHS = 5
ROBERTA_PATH = "../input/roberta-base"
ROBERTA_PATH_NEW = "../input/rb-base"

TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab_file=f"{ROBERTA_PATH}/vocab.json", 
    merges_file=f"{ROBERTA_PATH}/merges.txt", 
    lowercase=True,
    add_prefix_space=True
)


# In[ ]:


class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(ROBERTA_PATH, config=conf)
        self.drop_out = nn.Dropout(0.3)
        self.l0 = nn.Linear(768, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
    
    def forward(self, ids, mask, token_type_ids):
        seq, pooled = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        # out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.drop_out(seq)
        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


# In[ ]:


link_re = re.compile('http[s]?://\S+')
re_username = '^(\_)\w+'
re_username2 = '^@(\_)\w+'


def clean_text(text, sentiment):
    # cleaned_text = text.strip()
    cleaned_text = " ".join(str(text).split()).strip()
    if sentiment != 'neutral':
        if '_it_good' in text or '_in_love' in text or '_violence' in text:
            return text

        if re.search(re_username, cleaned_text):
            cleaned_text = re.sub(re_username, '', cleaned_text).strip()
        if re.search(re_username2, cleaned_text):
            cleaned_text = re.sub(re_username2, '', cleaned_text).strip()
    
        if re.search(link_re, cleaned_text): 
            split_text = text.split()
            if re.search(link_re, split_text[0]) or re.search(link_re, split_text[-1]):
                cleaned_text = re.sub('http[s]?://\S+', '', text).strip()

        if cleaned_text.split()[0] == '-':
            cleaned_text = cleaned_text[1:].strip()

        if 'these dogs are going to die if somebody doesn`t save them!' in cleaned_text:
            cleaned_text = 'these dogs are going to die if somebody doesn`t save them!'
        elif cleaned_text == 'imo':
            cleaned_text = 'Thanks'
        elif cleaned_text == 'and_jay hi! **** your job!':
            cleaned_text = 'hi! **** your job!'
        elif cleaned_text == 'beckett Thanks':
            cleaned_text = 'Thanks'
        elif 'c Thank' in cleaned_text:
            cleaned_text = 'Thank' 
        elif cleaned_text == 'hall no i was gutted when he wasn`t. lmao. i think i`m obsessed with him, bahaha.':
            cleaned_text = 'no i was gutted when he wasn`t. lmao. i think i`m obsessed with him, bahaha.'

    # print('Cleaned', cleaned_text)

    return cleaned_text


# In[ ]:


def process_data(tweet, selected_text, sentiment, tokenizer, max_len):
    
    tweet = " " + " ".join(str(tweet).split())
    selected_text = " " + " ".join(str(selected_text).split())
    
    
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
        'positive': [1313],
        'negative': [2430],
        'neutral': [7974]
    }

    
    input_ids = [0] + sentiment_id[sentiment] + [2] + [2] + input_ids_orig + [2]
    token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
    targets_start += 4
    targets_end += 4


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
        'targets_start': targets_start,
        'targets_end': targets_end,
        'orig_tweet': tweet,
        'orig_selected': selected_text,
        'sentiment': sentiment,
        'offsets': tweet_offsets
    }


# In[ ]:



class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN
    
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
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
            'orig_tweet': data["orig_tweet"],
            'orig_selected': data["orig_selected"],
            'sentiment': data["sentiment"],
            'offsets': torch.tensor(data["offsets"], dtype=torch.long)
        }


# In[ ]:


punc = "!."
def postprocess(text, predicted_text):
    # print(text)
    # print(predicted_text)
    splitted = text.split(predicted_text)[0]
    sub = len(splitted) - len(" ".join(splitted.split()))
    
#     ## new condition
#     if sub  == 1 and text.strip() != predicted_text.strip() and text[0] == " ":
#         start = text.find(predicted_text)
#         end = start + len(predicted_text)
#         start = start - 2 if start > 1 and text[start - 1] == " " else start - 1
#         end = end if (len(text) > end and text[end] == " ") or len(text) == end else end - 1
#         predicted_text = text[start: end]

    splitted1 = text.split(predicted_text.strip())[0]
    sub1 = len(splitted1) - len(" ".join(splitted1.split()))
    
    if sub1 == 2 and text.strip() != predicted_text.strip() and text[:2] == "  ":
        predicted_text = predicted_text.strip()
        start = text.find(predicted_text)
        end = start + len(predicted_text)
        if start == 0:
            end = end - 2
        else:
            start = start - 2
        if predicted_text[-1] in "!.,":
            end = end - 1
        predicted_text = text[start: end]

    
    elif sub > 1 and text.strip() != predicted_text.strip():
        if len(predicted_text.split()) == 1:
            if text[0] == " ":
                text = text[1:] + " "
            else:
                predicted_text = predicted_text + " "
            if text[0] == " " and text.strip().find(predicted_text.strip()) == 0:
                text = " " + text[:-1]
            add_one = False
            if predicted_text[-1] not in punc:
                add_one = True
            splitted = text.split(predicted_text)[0]
            sub = len(splitted) - len(" ".join(splitted.split()))
            start = text.find(predicted_text) - sub
            if start < 0:
                start = text.find(predicted_text.strip())
                add_one = False
                predicted_text = predicted_text.strip()
            if add_one:
                end = start + len(predicted_text) + 1
            else:
                end = start + len(predicted_text)
            predicted_text = text[start: end]
        
#         # start condition
#         elif len(predicted_text.split()) > 1:
#             splitted1 = text.split(predicted_text)
#             if len(splitted1) == 1 or splitted1[1] == "":
#                 start = text.find(predicted_text.strip()) - sub
#                 end = len(text) + 1
#             else:
#                 # shift end index as well
#                 start = text.find(predicted_text.strip()) - sub
#                 end = start + len(predicted_text) + 1

#             if start < 0:
#                 start = 0
#             predicted_text = text[start: end]  

    return predicted_text


# In[ ]:


# punc = "!."
# def postprocess(text, predicted_text):
#     # print(text)
#     # print(predicted_text)
#     splitted = text.split(predicted_text)[0]
#     sub = len(splitted) - len(" ".join(splitted.split()))
#     if sub > 1 and text.strip() != predicted_text.strip():
#         if len(predicted_text.split()) == 1:
#             if text[0] == " ":
#                 text = text[1:] + " "
#             else:
#                 predicted_text = predicted_text + " "
#             if text[0] == " " and text.strip().find(predicted_text.strip()) == 0:
#                 text = " " + text[:-1]
#             add_one = False
#             if predicted_text[-1] not in punc:
#                 add_one = True
#             splitted = text.split(predicted_text)[0]
#             sub = len(splitted) - len(" ".join(splitted.split()))
#             start = text.find(predicted_text) - sub
#             if start < 0:
#                 start = text.find(predicted_text.strip())
#                 add_one = False
#                 predicted_text = predicted_text.strip()
#             if add_one:
#                 end = start + len(predicted_text) + 1
#             else:
#                 end = start + len(predicted_text)
#             predicted_text = text[start: end]
#     return predicted_text


# In[ ]:


def calculate_jaccard_score(
    original_tweet, 
    target_string, 
    sentiment_val, 
    idx_start, 
    idx_end, 
    offsets,
    verbose=False):
    
    if idx_end < idx_start:
        idx_end = idx_start
    
    filtered_output  = ""
    for ix in range(idx_start, idx_end + 1):
        filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
            filtered_output += " "

#     if sentiment_val == "neutral" or len(original_tweet.split()) < 2:
#         filtered_output = original_tweet
        
    if len(original_tweet.split()) < 2:
        filtered_output = original_tweet
        
        
#     if len(filtered_output.strip()) > 0:
#         filtered_output = postprocess(original_tweet, filtered_output)
    
    if sentiment_val != "neutral" and verbose == True:
        if filtered_output.strip().lower() != target_string.strip().lower():
            print("********************************")
            print(f"Output= {filtered_output.strip()}")
            print(f"Target= {target_string.strip()}")
            print(f"Tweet= {original_tweet.strip()}")
            print("********************************")

    jac = 0
    return jac, filtered_output


# In[ ]:


df_test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
df_test.loc[:, "selected_text"] = df_test.text.values


# In[ ]:


# df_test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")


# In[ ]:


device = torch.device("cuda")
model_config = transformers.RobertaConfig.from_pretrained(ROBERTA_PATH)
#model_config = transformers.RobertaConfig.from_pretrained(ROBERTA_PATH)
#model_config.output_hidden_states = True


# In[ ]:


MODEL_BASE_PATH_OLD = '../input/roberta-base-uncased4'

MODEL_BASE_PATH = '../input/roberta-pp2'


# In[ ]:


ENSEMBLES = [
    {'model': TweetModel(conf=model_config), 'state_dict': f"{MODEL_BASE_PATH}/model_0.bin", 'weight': 1}, # CV: 0.705 LB:?? 0.709
    {'model': TweetModel(conf=model_config), 'state_dict': f"{MODEL_BASE_PATH}/model_1.bin", 'weight': 1}, # CV: 0.707 LB:?? 0.710
    {'model': TweetModel(conf=model_config), 'state_dict': f"{MODEL_BASE_PATH}/model_2.bin", 'weight': 1}, # CV: 0.712 LB:??
    {'model': TweetModel(conf=model_config), 'state_dict': f"{MODEL_BASE_PATH}/model_3.bin", 'weight': 1}, # CV: 0.718 LB:??
    {'model': TweetModel(conf=model_config), 'state_dict': f"{MODEL_BASE_PATH}/model_4.bin", 'weight': 1}, # CV: 0.718 LB:??
    {'model': TweetModel(conf=model_config), 'state_dict': f"{MODEL_BASE_PATH}/model_5.bin", 'weight': 1}, # CV: 0.715 LB:??
    {'model': TweetModel(conf=model_config), 'state_dict': f"{MODEL_BASE_PATH}/model_6.bin", 'weight': 1}, # CV: 0.710 LB:??
    {'model': TweetModel(conf=model_config), 'state_dict': f"{MODEL_BASE_PATH}/model_7.bin", 'weight': 1}, # CV: 0.714 LB??
]


# In[ ]:


models = []
weights = []

for val in ENSEMBLES:
    model = val['model']
    model.to(device)
    model.load_state_dict(torch.load(val['state_dict']))
    model.eval()
    models.append(model)
    weights.append(val['weight'])


# In[ ]:


def get_best_start_end_idxs(_start_logits, _end_logits):
    best_logit = -1000
    best_idxs = None
    for start_idx, start_logit in enumerate(_start_logits):
        for end_idx, end_logit in enumerate(_end_logits[start_idx:]):
            logit_sum = (start_logit + end_logit).item()
            if logit_sum > best_logit:
                best_logit = logit_sum
                best_idxs = (start_idx, start_idx+end_idx)
    return best_idxs


# In[ ]:


final_output = []


test_dataset = TweetDataset(
    tweet=df_test.text.values,
    sentiment=df_test.sentiment.values,
    selected_text=df_test.selected_text.values
)

data_loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=VALID_BATCH_SIZE,
    num_workers=1
)

# pp_model = pickle.load(open('../input/tse-post-process/post_process.sav', 'rb'))


with torch.no_grad():
    tk0 = tqdm(data_loader, total=len(data_loader))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_tweet = d["orig_tweet"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        offsets = d["offsets"].numpy()

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)

        outputs_start = []
        outputs_end = []
        
        for index, model in enumerate(models):
            output_start, output_end = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            outputs_start.append(output_start * weights[index]) 
            outputs_end.append(output_end * weights[index]) 
            
        outputs_start = sum(outputs_start) / len(outputs_start) 
        outputs_end = sum(outputs_end) / len(outputs_end)
    
        
        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
        jaccard_scores = []
        for px, tweet in enumerate(orig_tweet):
            selected_tweet = orig_selected[px]
            tweet_sentiment = sentiment[px]
            idx_start, idx_end = get_best_start_end_idxs(outputs_start[px, :], outputs_end[px, :]) 
            
            _, output_sentence = calculate_jaccard_score(
                original_tweet=tweet,
                target_string=selected_tweet,
                sentiment_val=tweet_sentiment,
#                 idx_start=np.argmax(outputs_start[px, :]),
#                 idx_end=np.argmax(outputs_end[px, :]),
                idx_start=idx_start,
                idx_end=idx_end,
                offsets=offsets[px]
            )
            
            final_output.append(output_sentence)


# In[ ]:


df_test['selected_text'] = final_output
df_test['selected_text'] = df_test.apply(lambda x: postprocess(x.text, x.selected_text), axis=1)


# In[ ]:


def place_in_back(x):
    splitted = x.text.split(x.selected_text)[0]
    sub = len(splitted) - len(" ".join(splitted.split()))
    
    select=x.selected_text
    ind = x.text.find(x.selected_text.strip())-1
    if ((ind >0) & (select.startswith('.')!=1) & (sub>0)):
        if ((x.text[ind] in string.punctuation) & (x.sentiment!='neutral') & (sub==1)):
            select = x.text[ind]+select
            print(select,sub)
        elif ((x.text[ind-1] in string.punctuation) & (x.sentiment!='neutral') &(sub==2) ):
            select = x.text[ind-1]+""+select
            print(select,sub)
    return select
    


# In[ ]:


df_test['selected_text']=df_test.apply(lambda x : place_in_back(x),axis=1)


# In[ ]:



sample = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
sample.loc[:, 'selected_text'] = df_test['selected_text']



sample['selected_text'] = sample['selected_text'].apply(lambda x: x.replace('!!!', '!') if len(x.split())==1 else x)
sample['selected_text'] = sample['selected_text'].apply(lambda x: x.replace('!!!!', '!!') if len(x.split())==1 else x)
sample['selected_text'] = sample['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split())==1 else x)
sample['selected_text'] = sample['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split())==1 else x)
sample['selected_text'] = sample['selected_text'].apply(lambda x: x.replace('....', '..') if len(x.split())==1 else x)




sample.to_csv("submission.csv", index=False)


# In[ ]:


sample.sample(10)


# In[ ]:




