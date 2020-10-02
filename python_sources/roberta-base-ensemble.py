#!/usr/bin/env python
# coding: utf-8

# # Load packages and scripts

# In[ ]:


# basic packages
import os, argparse
import json
import shutil
import warnings
import time
import psutil
from pathlib import Path
import tqdm
import numpy as np
import pandas as pd
import re, gc
from typing import Dict
from collections import OrderedDict, defaultdict

# torch related
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
# transformers & tokenizers
import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, AutoConfig, AutoModel, AutoTokenizer
import tokenizers

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

from shutil import copyfile

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore",category=UserWarning)


# # Model Inference Part I

# In[ ]:


copyfile(src = "../input/utils-v10/utilsv10.py", dst = "../working/utilsv10.py")
copyfile(src = "../input/utils-v10/dataset10.py", dst = "../working/dataset10.py")
copyfile(src = "../input/utils-v10/dataset11.py", dst = "../working/dataset11.py")

from utilsv10 import (binary_focal_loss, get_learning_rate, jaccard_list, get_best_pred, ensemble, ensemble_words,get_char_prob,
                   load_model, save_model, set_seed, write_event, evaluate, get_predicts_from_token_logits)

from dataset10 import TrainDataset, MyCollator
from dataset11 import TrainDataset as TrainDataset11


# ## Preapare data

# In[ ]:


df_pred = pd.read_csv('../input/tweet-sentiment-fast/test_all_post_finetune_0608.csv') #lb724
df_pred1 = pd.read_csv('../input/tweet-sentiment-fast/test_all_post_finetune_large.csv') #lb717
df_train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')

df_test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
#df_test = pd.read_csv('../input/tweet-sentiment-fast/test_hidden.csv')
df_test.loc[:, 'selected_text'] = df_test.text.values
df_test['text_clean'] = df_test['text'].apply(lambda x: " ".join(x.split()))

df_full = pd.read_csv('/kaggle/input/complete-tweet-sentiment-extraction-data/tweet_dataset.csv')
df_full = df_full[df_full.text.notnull()].copy()
df_full['text_clean'] = df_full['text'].apply(lambda x: " ".join(x.split()))
df_full = df_full.drop_duplicates(subset='text')
df_full = df_full[~df_full.aux_id.isin(df_train.textID)]
df_full.rename(columns={'sentiment': 'raw_sentiment'}, inplace=True)


# In[ ]:


def find_sentiment(textID, text, sentiment):
    text_clean = " ".join(text.split())
    if textID in df_full.aux_id.values:
        return df_full['raw_sentiment'].loc[df_full.aux_id==textID].values[0]
    elif text in df_full.text.values:
        return df_full['raw_sentiment'].loc[df_full.text==text].values[0]
    elif text_clean in df_full.text_clean.values:
        return df_full['raw_sentiment'].loc[df_full.text_clean==text_clean].values[0]
    else:
        return sentiment


# In[ ]:


#%%time
# find raw sentiment
df_test['raw_sentiment'] = df_test.apply(lambda x: find_sentiment(x['textID'], x['text'], x['sentiment']), axis=1)
na_mask = df_test.raw_sentiment.isnull()
print(na_mask.sum())


# ## Parse data

# In[ ]:


#test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
test = df_test.copy()
tokenizer = AutoTokenizer.from_pretrained('../input/roberta-base/', do_lower_case=False)

class Args:
    post = True
    tokenizer = tokenizer
    offset = 4
    batch_size = 4
    workers = 0
args = Args()

# v11
class Args11:
    post = True
    tokenizer = tokenizer
    offset = 7
    batch_size = 4
    workers = 0
args11 = Args11()


# ## Model

# In[ ]:


class TweetModel(nn.Module):

    def __init__(self, pretrain_path=None, dropout=0.2, config=None):
        super(TweetModel, self).__init__()
        if config is not None:
            self.bert = AutoModel.from_config(config)
        else:
            config = AutoConfig.from_pretrained(pretrain_path, output_hidden_states=True)
            self.bert = AutoModel.from_pretrained(
                pretrain_path, cache_dir=None, config=config)
        
        self.cnn =  nn.Conv1d(self.bert.config.hidden_size*3, self.bert.config.hidden_size, 3, padding=1)

        # self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size//2, num_layers=2,
        #                     batch_first=True, bidirectional=True)
        self.gelu = nn.GELU()

        self.whole_head = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(0.1)),
            ('l1', nn.Linear(self.bert.config.hidden_size*3, 256)),
            ('act1', nn.GELU()),
            ('dropout', nn.Dropout(0.1)),
            ('l2', nn.Linear(256, 2))
        ]))
        self.se_head = nn.Linear(self.bert.config.hidden_size, 2)
        self.inst_head = nn.Linear(self.bert.config.hidden_size, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs, masks, token_type_ids=None, input_emb=None):
        _, pooled_output, hs = self.bert(
            inputs, masks, token_type_ids=token_type_ids, inputs_embeds=input_emb)

        seq_output = torch.cat([hs[-1],hs[-2],hs[-3]], dim=-1)

        # seq_output = hs[-1]

        avg_output = torch.sum(seq_output*masks.unsqueeze(-1), dim=1, keepdim=False)
        avg_output = avg_output/torch.sum(masks, dim=-1, keepdim=True)
        # +max_output
        whole_out = self.whole_head(avg_output)

        seq_output = self.gelu(self.cnn(seq_output.permute(0,2,1)).permute(0,2,1))
        
        se_out = self.se_head(self.dropout(seq_output))  #()
        inst_out = self.inst_head(self.dropout(seq_output))
        return whole_out, se_out[:, :, 0], se_out[:, :, 1], inst_out


# In[ ]:


def predict_wu(model: nn.Module, valid_df, valid_loader, args, progress=False) -> Dict[str, float]:
    # run_root = Path('../experiments/' + args.run_root)
    model.eval()
    all_end_pred, all_whole_pred, all_start_pred, all_inst_out = [], [], [], []
    if progress:
        tq = tqdm.tqdm(total=len(valid_df))
    with torch.no_grad():
        for tokens, types, masks, _, _, _, _, _, _, _ in valid_loader:
            if progress:
                batch_size = tokens.size(0)
                tq.update(batch_size)
            masks = masks.cuda()
            tokens = tokens.cuda()
            types = types.cuda()
            whole_out, start_out, end_out, inst_out = model(tokens, masks, types)
            
            all_whole_pred.append(torch.softmax(whole_out, dim=-1)[:,1].cpu().numpy())
            inst_out = torch.softmax(inst_out, dim=-1)
            for idx in range(len(start_out)):
                length = torch.sum(masks[idx,:]).item()-1 # -1 for last token
                all_start_pred.append(torch.softmax(start_out[idx, args.offset:length], axis=-1).cpu())
                all_end_pred.append(torch.softmax(end_out[idx, args.offset:length], axis=-1).cpu())
                all_inst_out.append(inst_out[idx,:,1].cpu())
            assert all_start_pred[-1].dim()==1

    all_whole_pred = np.concatenate(all_whole_pred)
    
    if progress:
        tq.close()
    return all_whole_pred, all_start_pred, all_end_pred, all_inst_out


# ## Predict

# In[ ]:


# convrt part I char-level probability to clean text version (an array of length 160)
def _convrt_prob_partI(text, prob, max_char_len=160):
    clean_text = " ".join(text.split())
    new_prob = []
    p1, p2 = 0, 0
    while p1 < len(text):
        if text[p1] not in [" ", "\t", "\xa0"]:
            if text[p1] == clean_text[p2]:
                new_prob.append(prob[p1])
                p1 += 1
                p2 += 1
        else:
            if p1 + 1 < len(text) and text[p1+1] not in [" ", "\t", "\xa0"]:
                if clean_text[p2] == " ":
                    new_prob.append(prob[p1])
                    p1 += 1
                    p2 += 1
                else:
                    p1 += 1                   
            else:
                p1 += 1
    if len(new_prob) < max_char_len:
        new_prob = new_prob + (max_char_len - len(new_prob))*[0]
    return new_prob


# In[ ]:


def get_prediction_partI(weights_path, test, test_loader, args, output_name):
    # load model
    config = RobertaConfig.from_pretrained('../input/roberta-base', output_hidden_states=True)
    model = TweetModel(config=config)

    # 10-fold predict
    all_whole_preds, all_start_preds, all_end_preds, all_inst_preds = [], [], [], []  
    for fold in range(10):
        load_model(model, f'{weights_path}/best-model-%d.pt' % fold)
        model.cuda()
        fold_whole_preds, fold_start_preds, fold_end_preds, fold_inst_preds = predict_wu(model, test, test_loader, args, progress=True)

        all_whole_preds.append(fold_whole_preds)
        all_start_preds.append(fold_start_preds)
        all_end_preds.append(fold_end_preds)
        all_inst_preds.append(fold_inst_preds)


    all_whole_preds, all_start_preds, all_end_preds, all_inst_preds = ensemble(all_whole_preds, all_start_preds, all_end_preds, all_inst_preds, test)
    word_preds, inst_word_preds, scores = get_predicts_from_token_logits(all_whole_preds, all_start_preds, all_end_preds, all_inst_preds, test, args)
    # word_preds, inst_word_preds, scores = get_predicts_from_token_logits(fold_whole_preds, fold_start_preds, fold_end_preds, fold_inst_preds, test, args)
    start_char_prob, end_char_prob = get_char_prob(all_start_preds, all_end_preds, test, args)
    
    test['start_char_prob'] = start_char_prob
    test['end_char_prob'] = end_char_prob
    test['selected_text'] = word_preds

    test['prob_start'] = test.apply(lambda x: _convrt_prob_partI(x['text'], x['start_char_prob']), axis=1)
    test['prob_end'] = test.apply(lambda x: _convrt_prob_partI(x['text'], x['end_char_prob']), axis=1)

    test.to_pickle(f'{output_name}.pkl')
    np.save(f"start_{output_name}.npy", np.array(test['prob_start'].tolist()))
    np.save(f"end_{output_name}.npy", np.array(test['prob_end'].tolist()))


# ### V10 prediction

# In[ ]:


collator = MyCollator()
test_set = TrainDataset(test, None, tokenizer=tokenizer, mode='test', offset=args.offset)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
                         num_workers=args.workers)
get_prediction_partI(weights_path='../input/roberta-v10-10',
                     test=test, 
                     test_loader=test_loader, 
                     args=args, 
                     output_name='output_v10',
                    )

mem = psutil.virtual_memory()
print(f'{mem.percent:5} - {mem.free/1024**3:10.2f} - {mem.available/1024**3:10.2f} - {mem.used/1024**3:10.2f}')


# ### V11 prediction

# In[ ]:


collator = MyCollator()
test_set = TrainDataset11(test, None, tokenizer=tokenizer, mode='test', offset=args11.offset)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
                         num_workers=args.workers)
get_prediction_partI(weights_path='../input/roberta-v11-10',
                     test=test, 
                     test_loader=test_loader, 
                     args=args11, 
                     output_name='output_v11',
                    )

mem = psutil.virtual_memory()
print(f'{mem.percent:5} - {mem.free/1024**3:10.2f} - {mem.available/1024**3:10.2f} - {mem.used/1024**3:10.2f}')


# # Model Inference Part II

# In[ ]:


# import helper scripts
from shutil import copyfile

# copy our file into the working directory (make sure it has .py suffix)
copyfile(src = "../input/tweet-sentiment/common.py", dst = "../working/common.py")
copyfile(src = "../input/tweet-sentiment/dataset.py", dst = "../working/dataset.py")
copyfile(src = "../input/tweet-sentiment/models.py", dst = "../working/models.py")
copyfile(src = "../input/tweet-sentiment/metrics.py", dst = "../working/metrics.py")
copyfile(src = "../input/tweet-sentiment/utils.py", dst = "../working/utils.py")
copyfile(src = "../input/tweet-sentiment/predict_fn.py", dst = "../working/predict_fn.py")
copyfile(src = "../input/tweet-sentiment/nlp_albumentations.py", dst = "../working/nlp_albumentations.py")
copyfile(src = "../input/tweet-sentiment/transform.py", dst = "../working/transform.py")
copyfile(src = "../input/tweet-sentiment/run_inference_kaggle.py", dst = "../working/run_inference_kaggle.py")

from dataset import process_data
from dataset import TweetDataset_kaggle as TweetDataset
from models import TweetModel, TweetModel_v2
from common import *
from metrics import *
from utils import *
from utils import _convrt_back

set_seed(42)

# # %% [code]
# !python run_inference_kaggle.py --model_name='roberta_base' \
#                                 --model_path='../input/roberta-base/' \
#                                 --raw_sentiment=1

# mem = psutil.virtual_memory()
# print(f'{mem.percent:5} - {mem.free/1024**3:10.2f} - {mem.available/1024**3:10.2f} - {mem.used/1024**3:10.2f}')

# # %% [code]
# !python run_inference_kaggle.py --model_name='roberta_base_noRawSenti' \
#                                 --model_path='../input/roberta-base-bs32-v2-0608' \
#                                 --raw_sentiment=0

# mem = psutil.virtual_memory()
# print(f'{mem.percent:5} - {mem.free/1024**3:10.2f} - {mem.available/1024**3:10.2f} - {mem.used/1024**3:10.2f}')


# ## Helper functions & params

# In[ ]:


def get_model(model_name):
    # return model and tokenizer
    if model_name.startswith('roberta_base'):
        model_info = {
            'name': 'roberta-base',
            'model_path': '../input/roberta-base' if ON_KAGGLE else 'roberta-base',
            'from_pretrained': False if ON_KAGGLE else True,
            'vocab_file': '../input/roberta-base/vocab.json',
            'merges_file': '../input/roberta-base/merges.txt',
        }
    elif model_name.startswith('roberta_large'):
        model_info = {
            'name': 'roberta-large',
            'model_path': '../input/roberta-large' if ON_KAGGLE else 'roberta-large',
            'from_pretrained': False if ON_KAGGLE else True,
            'vocab_file': '../input/roberta-large/vocab.json',
            'merges_file': '../input/roberta-large/merges.txt',
        }
    else:
        raise RuntimeError('%s is not implemented.' % model_name)

    model = TweetModel_v2(model_info)
    tokenizer = tokenizers.ByteLevelBPETokenizer(
        vocab_file=model_info['vocab_file'],
        merges_file=model_info['merges_file'],
        lowercase=False,
        add_prefix_space=True
    )
    return model, tokenizer


# ## Dataloader, model

# In[ ]:


N_FOLD = 10
POST_PROCESS = True
params = {
    'models': [
        #'roberta_base', 
        'roberta_base_noRawSenti', 
        'roberta_large',
    ],
    'batch_size': 4,
    'workers': 1 if ON_KAGGLE else 8,
    'max_len': 192, 
    'folds': list(x for x in range(N_FOLD)),
    'limit': 0,
}

path_lib = {
    'roberta_base': '../input/roberta-base/',
    'roberta_base_noRawSenti': '../input/roberta-base-bs32-v2-0608',
    'roberta_large': '../input/roberta-large-bs32-v2-0608',
}


# ## Prediction

# In[ ]:


def predict_II(model, data_loader, tokenizer):
    start_probs, end_probs = [], []
    with torch.no_grad():
        for i, d in enumerate(tqdm.tqdm(data_loader, ascii=True)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            decode_selected = d["decode_selected"]
            raw_tweets = d["raw_tweet"]
            raw_selecteds = d["raw_selected_text"]
            text_span = d["text_span"]
            
            ids = ids.cuda()
            token_type_ids = token_type_ids.cuda()
            mask = mask.cuda()
            
            start_idxs, end_idsx = [], []
            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            if len(outputs) == 2:
                outputs_start, outputs_end = outputs
                outputs_mask = None
            elif len(outputs) == 3:
                outputs_start, outputs_end, outputs_mask = outputs   
            # probability
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy() 
            
            for px, tweet in enumerate(orig_tweet):
                # char level probs
                raw_tweet = raw_tweets[px]
                span_start, span_end = text_span[0][px], text_span[1][px]
                span_start, span_end = int(span_start), int(span_end)
                token_ids = d["ids"][px][span_start: span_end]
                clean_text = " ".join(raw_tweet.split())
                _, prob_char = convrt_prob_char_level(clean_text, token_ids, outputs_start[px, span_start: span_end], tokenizer)
                if len(prob_char) < 160: # padding
                    prob_char += [0] * (160 - len(prob_char))
                start_probs.append(prob_char)
                _, prob_char = convrt_prob_char_level(clean_text, token_ids, outputs_end[px, span_start: span_end], tokenizer)
                if len(prob_char) < 160: # padding
                    prob_char += [0] * (160 - len(prob_char))
                end_probs.append(prob_char)
    start_probs = np.array(start_probs)
    end_probs = np.array(end_probs)
    return start_probs, end_probs


# In[ ]:


# predict
for model_name in params['models']:
    model, tokenizer = get_model(model_name)
    if model_name.endswith("noRawSenti"):
        df_test_tmp = df_test.copy()
        df_test_tmp['raw_sentiment'] = ""
        tweet_dataset = TweetDataset(
            df=df_test_tmp,
            sentiment_weights=[1,1,1],
            tokenizer=tokenizer,
            mode='test',
            lower_case=0,
            max_len=params['max_len'],
        )
    else:
        tweet_dataset = TweetDataset(
            df=df_test,
            sentiment_weights=[1,1,1],
            tokenizer=tokenizer,
            mode='test',
            lower_case=0,
            max_len=params['max_len'],
        )      
        
    data_loader = DataLoader(
        tweet_dataset,
        batch_size=params['batch_size'],
        num_workers=0,
    )
        
    for i, fold in enumerate(params['folds']):
        if model_name == "roberta_base_noRawSenti":
            if fold < 5:
                path = path_lib[model_name] + '-part2'
            else:
                path = path_lib[model_name]
        elif model_name == "roberta_large":
            if fold <= 2:
                path = path_lib[model_name] + '-part1'
            elif fold <= 5:
                path = path_lib[model_name] + '-part2'
            elif fold == 6:
                path = path_lib[model_name] + '-part3-2'
            elif fold == 7:
                path = path_lib[model_name] + '-part3'
            else:
                path = path_lib[model_name] + '-part4'
        else:
            path = path_lib[model_name]
        load_model(model, f"{path}/best_jac_{fold}.pt", multi2single=False)
        model.cuda()
        model.eval()
        probs_start_pred, probs_end_pred = predict_II(model, data_loader, tokenizer)
        if i == 0:
            probs_start = probs_start_pred
            probs_end = probs_end_pred
        else:
            probs_start += probs_start_pred
            probs_end += probs_end_pred 
            
    probs_start /= len(params['folds'])
    probs_end /= len(params['folds'])

    df_test['prob_start'] = probs_start.tolist()
    df_test['prob_end'] = probs_end.tolist()
    df_test.to_pickle(f'{model_name}.pkl')
    np.save(f"start_{model_name}.npy", probs_start)
    np.save(f"end_{model_name}.npy", probs_end)


# # Ensemble of different models

# In[ ]:


# load .npy file from disk and ensemble char level probability

model_names = [
               #'roberta_base', 
               'roberta_base_noRawSenti',
               'roberta_large',
               'output_v10',
               'output_v11',
              ]
for i, model_name in enumerate(model_names):
    prob_start_tmp = np.load(f'start_{model_name}.npy')
    prob_end_tmp = np.load(f'end_{model_name}.npy')
    if i == 0:
        prob_start = prob_start_tmp
        prob_end = prob_end_tmp
    else:
        prob_start += prob_start_tmp
        prob_end += prob_end_tmp

prob_start /= len(model_names)
prob_end /= len(model_names)
        
df_test['prob_start'] = prob_start.tolist()
df_test['prob_end'] = prob_end.tolist()


# # Post processing

# In[ ]:


def _get_pred_char(df, probs_start, probs_end):
    df['start_idx'] = np.argmax(probs_start, axis=1)
    df['end_idx'] = probs_end.shape[1] - np.argmax(probs_end[:, ::-1], axis=1) - 1
    df['prob_start'] = probs_start.tolist()
    df['prob_end'] = probs_end.tolist()
    idxs = np.where(df.start_idx > df.end_idx)
    
    for idx in idxs[0]:
        prob_start = df.prob_start.values[idx]
        prob_end = df.prob_end.values[idx]
        start_idx = df.start_idx.values[idx]
        end_idx = df.end_idx.values[idx]
        if prob_start[start_idx] > prob_end[end_idx] or end_idx == 0:
            end_idx = len(prob_start) - np.argmax(prob_end[start_idx:][::-1]) - 1
        else:
            start_idx = np.argmax(prob_start[:end_idx])
        df['start_idx'].iloc[idx] = start_idx     
        df['end_idx'].iloc[idx] = end_idx    
    #df.rename(columns={'selected_text': 'pred'}, inplace=True)
    df['pred_char'] = df.apply(lambda x: x['text_clean'][x['start_idx']: x['end_idx']+1], axis=1)
    return df


# In[ ]:


def post_neutral(df):
    df['select_pt'] = df.apply(lambda x: len(x['pred_char'].strip())/len(x['text_clean']), axis=1).values

    raw_sents = ['neutral', 'sadness', 'worry', 'happiness', 'love', 'enthusiasm']
    mm = (df['sentiment'] == 'neutral') & (df['raw_sentiment'].isin(raw_sents))
    mm = (mm |          ((df.select_pt > 0.85) & (df.sentiment.isin(['positive']))) |          ((df.select_pt > 0.85) & (df.sentiment.isin(['negative']))) |          ((df.select_pt < 0.2) & (df.sentiment.isin(['neutral']))))
    
    df['pred_exp'] = df['pred_char'].values
    df['pred_exp'].loc[mm] = df['text_clean'].loc[mm].values
    print(f"# of modified samples: {np.sum(df['pred_exp'] != df['pred_char'])}")
    return df

def _post_shift_new(text, pred):
    clean_text = " ".join(text.split())
    start_clean = clean_text.find(" ".join(pred.split()))
    
    raw_pred = _convrt_back(text, pred, "neutral")
    raw_pred = raw_pred.strip()
    start = text.find(raw_pred)
    end = start + len(raw_pred)
    
    extra_space = start - start_clean 
    
    if start>extra_space and extra_space>0:
        if extra_space==1:
            if text[start-1] in [',','.','?','!'] and text[start-2]!=' ':
                start -= 1
        elif extra_space==2:
            start -= extra_space
            if text[end-1] in [',','.','!','?','*']:
                end -= 1
        else:
            end -= (extra_space-2)
            start -= extra_space
    
    pred = text[start:end]
    if pred.count("'") == 1:
        if pred[0] == "'":
            if text.find(pred) + len(pred) < len(text) and text[text.find(pred) + len(pred)] == "'":
                pred += "'"
        else:
            if text.find(pred) - 1 >= 0 and text[text.find(pred) - 1] == "'":
                pred = "'" + pred   
                
    return pred

def post_shift(df):
    df['pred_final'] = df['pred_exp'].copy()
    df['jac_text'] = df.apply(lambda x: jaccard(x['text'], x['pred_exp']), axis=1)
    mask = (df.sentiment != 'neutral') & (df.start_idx != 0) & (df.jac_text != 1)
    df['pred_final'].loc[mask] = df.apply(lambda x: _post_shift_new(x['text'], x['pred_exp']), axis=1).loc[mask].values
    jac = df.apply(lambda x: jaccard(x['pred_exp'], x['pred_final']), axis=1)
    print(f"# of modified samples: {np.sum(jac != 1)}")
    return df


# In[ ]:


# _get_pred_char bug ==> lb 717 (pred1)
# POST_PROCESS bug ==> lb 0
POST_PROCESS=True
try:
    df_test = _get_pred_char(df_test, np.array(df_test.prob_start.tolist()), np.array(df_test.prob_end.tolist()))
    try:
        if POST_PROCESS:
            df_test = post_neutral(df_test)
            df_test = post_shift(df_test)
            df_test['selected_text'] = df_test['pred_final'].values
        else:
            df_test['selected_text'] = df_test['pred_char'].values
    except:
        df_test['selected_text'] = " "
except:
    df_test = pd.merge(df_test[['textID']], df_pred1, how='left', on='textID')


# # Save prediction

# In[ ]:


df_test.to_csv("raw_prediction.csv", index=False)
df_sub = df_test[['textID', 'selected_text']].copy()
df_sub.to_csv("submission.csv", index=False)


# In[ ]:


df_sub.head()


# In[ ]:




