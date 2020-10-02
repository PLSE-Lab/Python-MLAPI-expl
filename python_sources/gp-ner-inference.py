#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install pytorch-transformers')


# In[ ]:


import os
import json
from operator import itemgetter
from itertools import groupby
from tqdm import tqdm_notebook as tqdm
import textwrap
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_transformers import *
import torch.utils.data as d


# In[ ]:


class BertForNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForNER, self).__init__(config)
        self.num_labels = 3

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits


# In[ ]:


class TrainedModel:
    def __init__(self, model_dir, max_len, batch_size, device=None, model=None, tokenizer=None):
        self.max_len = max_len
        self.batch_size = batch_size
        
        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model is None and tokenizer is None:
            self.tokenizer, self.model = self._init_model(model_dir)
        else:
            self.tokenizer, self.model = tokenizer, model
        self.model.eval()
        
    def _init_model(self, model_dir):
        tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=False)
        model = BertForNER            .from_pretrained(model_dir)            .to(self.device)
        return tokenizer, model
    
    def _text_to_sequences(self, data):
        max_seq_length = self.max_len - 2
        all_tokens = []
        longer = 0
       
        for item in data:
            tokens = self.tokenizer.tokenize(item)

            if len(tokens) > max_seq_length:
                tokens = tokens[:max_seq_length]
                longer += 1

            final_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"]) +                 [0] * (max_seq_length - len(tokens))

            all_tokens.append(final_tokens)

        return np.array(all_tokens)
    
    def _get_test_loader(self, tokens):
        x_torch = torch.tensor(tokens, dtype=torch.long).to(self.device)

        dataset = d.TensorDataset(x_torch)
        sampler = d.SequentialSampler(dataset)
        dataloader = d.DataLoader(dataset,
                                                 sampler=sampler, 
                                                 batch_size=self.batch_size)
        return dataloader
    
    def _get_ranges(self, lst):
        pos = (j - i for i, j in enumerate(lst))
        t = 0
        for i, els in groupby(pos):
            l = len(list(els))
            el = lst[t]
            t += l
            yield list(range(el, el+l))
            
    def predict(self, texts, only_uniq=False):
        if type(texts) is str:
            texts = texts.split('.')

        sequences = self._text_to_sequences(texts)

        loader = self._get_test_loader(sequences)
        
        with torch.no_grad():
            preds = []
            for x_batch, in loader:
                y_pred = self.model(input_ids=x_batch, 
                                    attention_mask=(x_batch > 0).to(self.device))
                y_pred = y_pred.detach().cpu().numpy()

                for i in range(y_pred.shape[0]):
                    pred = y_pred[i]
                    ids = x_batch[i].detach().cpu().numpy()

                    nes = np.argmax(pred, axis=1)
                    pos_ids = np.where(nes > 0)[0]

                    ranges = list(self._get_ranges(pos_ids))

                    total_words = []
                    for token_ids in ranges:
                        ne_tokens = ids[token_ids]
                        ne_parts = self.tokenizer.convert_ids_to_tokens(ne_tokens)
                        if ne_parts[0][:2] == '##':
                            n = 0
                            while ne_parts[0][:2] == '##':
                                token_ids = [token_ids[0] - 1] + token_ids
                                ne_tokens = ids[token_ids]
                                ne_parts = self.tokenizer.convert_ids_to_tokens(ne_tokens)
                                n += 1
                                if n > 20:
                                    break
                        ne_words = self.tokenizer.convert_tokens_to_string(ne_parts)
                        total_words.append(ne_words)
                    preds.append(total_words)
        if only_uniq:
            preds = [item for items in preds for item in items]
            preds = list(set(preds))
        return preds


# In[ ]:


os.listdir('../input/pmc-articles')


# In[ ]:


articles_dir = '../input/pmc-articles/'
with open(os.path.join(articles_dir, 'gp_full.json')) as json_file:
    gp_full = json.load(json_file)
    
gp_articles = [item['article_id'] for item in gp_full]
len(gp_articles)


# In[ ]:


texts = [pd.read_csv(os.path.join(articles_dir, file))
         for file in os.listdir(articles_dir) if file.split('.')[-1] == 'csv']

texts = pd.concat(texts)
texts.columns = ['article_id', 'text']
texts.head()


# In[ ]:


texts = texts.loc[texts.article_id.isin(gp_articles)]
texts.shape[0]


# In[ ]:


texts = texts.loc[~texts.text.isna()]
texts.shape[0]


# In[ ]:


model_dir = '../input/scibert-uncased-pytorch/finetuned_diseases_ner'

tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=False)
model = BertForNER            .from_pretrained(model_dir)            .to('cuda')


# In[ ]:


max_len = 75
batch_size = 8

ner_model = TrainedModel(None, max_len, batch_size, model=model, tokenizer=tokenizer)


# In[ ]:


named_entitites = []


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nis_interactive = True\nif is_interactive:\n    iterator = tqdm(texts.iterrows(), total=texts.shape[0])\nelse:\n    iterator = texts.iterrows()\nfor i, row in iterator:\n    nes = ner_model.predict(row.text, only_uniq=True)\n    named_entitites.append({'article_id': row.article_id, 'entity': nes})\n    ")


# In[ ]:


entities = [e for e in named_entitites if e['entity'] != []]


# In[ ]:


with open('./entities.json', 'w') as outfile:
    json.dump(entities, outfile)

