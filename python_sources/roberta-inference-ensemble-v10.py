#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import json
import os
import random
import re
import shutil
from collections import OrderedDict, defaultdict
from functools import partial
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import tqdm


from sklearn.model_selection import GroupKFold
from torch import nn
from torch.nn import functional as F

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import (AdamW, get_cosine_schedule_with_warmup,
                                       get_linear_schedule_with_warmup,
                                       get_cosine_with_hard_restarts_schedule_with_warmup)


# In[ ]:


from shutil import copyfile
copyfile(src = "../input/utils-v10/utilsv10.py", dst = "../working/utilsv10.py")
copyfile(src = "../input/utils-v10/dataset10.py", dst = "../working/dataset10.py")


# In[ ]:


from utilsv10 import (binary_focal_loss, get_learning_rate, jaccard_list, get_best_pred, ensemble, ensemble_words,get_char_prob,
                   load_model, save_model, set_seed, write_event, evaluate, get_predicts_from_token_logits)


# In[ ]:


from dataset10 import TrainDataset, MyCollator


# ## Parse data

# In[ ]:


test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained('../input/roberta-base/', do_lower_case=False)


# In[ ]:


class Args:
    post = True
    tokenizer = tokenizer
    offset = 4
    batch_size = 32
    workers = 1
args = Args()


# ## dataset

# In[ ]:


collator = MyCollator()
test_set = TrainDataset(test, None, tokenizer=tokenizer, mode='test', offset=args.offset)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
                                 num_workers=args.workers)


# ## model

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


def predict(model: nn.Module, valid_df, valid_loader, args, progress=False) -> Dict[str, float]:
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


# ## predict

# In[ ]:


config = RobertaConfig.from_pretrained('../input/roberta-base', output_hidden_states=True)
model = TweetModel(config=config)


# In[ ]:


all_whole_preds, all_start_preds, all_end_preds, all_inst_preds = [], [], [], []

    
for fold in range(10):
    load_model(model, '../input/roberta-v10-10/best-model-%d.pt' % fold)
    model.cuda()
    fold_whole_preds, fold_start_preds, fold_end_preds, fold_inst_preds = predict(model, test, test_loader, args, progress=True)

    all_whole_preds.append(fold_whole_preds)
    all_start_preds.append(fold_start_preds)
    all_end_preds.append(fold_end_preds)
    all_inst_preds.append(fold_inst_preds)


all_whole_preds, all_start_preds, all_end_preds, all_inst_preds = ensemble(all_whole_preds, all_start_preds, all_end_preds, all_inst_preds, test)
word_preds, inst_word_preds, scores = get_predicts_from_token_logits(all_whole_preds, all_start_preds, all_end_preds, all_inst_preds, test, args)
# word_preds, inst_word_preds, scores = get_predicts_from_token_logits(fold_whole_preds, fold_start_preds, fold_end_preds, fold_inst_preds, test, args)
start_char_prob, end_char_prob = get_char_prob(all_start_preds, all_end_preds, test, args)


# In[ ]:


test['start_char_prob'] = start_char_prob
test['end_char_prob'] = end_char_prob


# In[ ]:


test['selected_text'] = word_preds
# test.loc[replace_idx, 'selected_text'] = test.loc[replace_idx, 'text']
def f(selected):
    return " ".join(set(selected.lower().split()))
# test.selected_text = test.selected_text.map(f)
test[['textID','selected_text']].to_csv('submission.csv', index=False)


# In[ ]:


test.head(20)


# In[ ]:




