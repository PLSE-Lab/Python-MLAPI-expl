#!/usr/bin/env python
# coding: utf-8

# # Preface

# Some non-engineer NLP beginners, like me, may be literate for PyTorch but not for Keras/TensorFlow.  
#   
# This notebook has two major objectives:
# 
# - To share starter code for PyTorch.
# 
# - To provide an example of easy implementation with PyTorch-Lightning. PyTorch does not provide wrappers for training neural networks, but PyTorch-Lightning will free us from explicit for-loops.

# # 0. Dependencies

# ### 0-0. Debug mode flag

# In[ ]:


DEBUG_MODE = False


# ### 0-1. Install PyTorch-Lightning

# This notebook requires PyTorch-Lighning.  
# 
# To use PyTorch-Lighning on Internet-off notebook, the following dataset is helpful:  
# 
# [pytorch-lightning 0.7.1](https://www.kaggle.com/higepon/pytorchlightning-071) by @[higepon](https://www.kaggle.com/higepon)

# In[ ]:


get_ipython().system('pip install ../input/pytorchlightning-071/pytorch-lightning-0.7.1/pytorch-lightning-0.7.1')


# ### 0-2. Import modules

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

import pytorch_lightning as pl

from tqdm import tqdm_notebook as tqdm


# # 1. Preparation

# ### 1-1. Ensure determinism

# In[ ]:


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# In[ ]:


seed_everything()


# ### 1-2. Load dataset

# In[ ]:


DIR = Path('../input/tweet-sentiment-extraction')


# In[ ]:


df_train = pd.read_csv(DIR / 'train.csv')
df_test = pd.read_csv(DIR / 'test.csv')


# ### 1-3. Clean dataset

# Convert a few float type samples in 'text' and 'selected_text' columns into strings.

# In[ ]:


df_train['text'] = df_train['text'].apply(lambda x: str(x))
df_test['text'] = df_test['text'].apply(lambda x: str(x))
df_train['uncased_text'] = df_train['text'].apply(lambda x: x.lower())
df_test['uncased_text'] = df_test['text'].apply(lambda x: x.lower())
df_train['selected_text'] = df_train['selected_text'].apply(lambda x: str(x).lower())


# ### 1-4. Tokenize

# Hugging Face Transformers tokenizers must be prepared in your Kaggle dataset to use in off-line notebook.  
#   
# See also [berttokenizer-base-uncased](https://www.kaggle.com/yutanakamura/berttokenizer-base-uncased).

# In[ ]:


tokenizer = BertTokenizer.from_pretrained('../input/berttokenizer-base-uncased')


# In[ ]:


# Tokenize
df_train['tokenized_text'] = df_train['uncased_text'].apply(tokenizer.tokenize)
df_test['tokenized_text'] = df_test['uncased_text'].apply(tokenizer.tokenize)
df_train['tokenized_selected_text'] = df_train['selected_text'].apply(tokenizer.tokenize)


# ### 1-5. Start & end positions

# Start positions and end positions of selected texts in tokenized source texts.

# In[ ]:


# Filter train data
start_position_candidates = []
end_position_candidates = []
df_train['select_length'] = df_train['tokenized_selected_text'].map(len)

for i in tqdm(range(len(df_train))):
    start_position_candidate = [j for j, tok in enumerate(df_train['tokenized_text'].iloc[i]) if tok == df_train['tokenized_selected_text'].iloc[i][0]]
    end_position_candidate = [j for j, tok in enumerate(df_train['tokenized_text'].iloc[i]) if tok == df_train['tokenized_selected_text'].iloc[i][-1]]

    start_position_candidate = [idx for idx in start_position_candidate if idx + df_train['select_length'].iloc[i] - 1 in end_position_candidate]
    end_position_candidate = [idx for idx in end_position_candidate if idx - df_train['select_length'].iloc[i] + 1 in start_position_candidate]

    start_position_candidates.append(start_position_candidate)
    end_position_candidates.append(end_position_candidate)


# If more than one candidates are available, we here take only the first candidate and discard the rest for plainness.

# In[ ]:


start_position_candidates = [l[0] if len(l) > 0 else -1 for l in start_position_candidates]
end_position_candidates = [l[0] if len(l) > 0 else -1 for l in end_position_candidates]


# For test data, set default start & end positions to dummy integer (-1).

# In[ ]:


df_train['start_position'] = start_position_candidates
df_train['end_position'] = end_position_candidates
df_test['start_position'] = -1
df_test['end_position'] = -1


# Drop train data with no candidates of start & end positions due to poor segmentation of selected texts.  
#   
# See also [this discussion](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/138272).

# In[ ]:


df_train = df_train.query('start_position!=-1')


# ### 1-6. Train/Val split

# In[ ]:


df_train, df_val = train_test_split(df_train, train_size=0.8)


# ### 1-7. Pos/Neg/Neu split

# This notebook will deal with positive, negative and neutral samples independently.

# In[ ]:


pos_train = df_train.query('sentiment=="positive"')
neg_train = df_train.query('sentiment=="negative"')
neu_train = df_train.query('sentiment=="neutral"')

pos_val = df_train.query('sentiment=="positive"')
neg_val = df_train.query('sentiment=="negative"')
neu_val = df_train.query('sentiment=="neutral"')

pos_test = df_test.query('sentiment=="positive"')
neg_test = df_test.query('sentiment=="negative"')
neu_test = df_test.query('sentiment=="neutral"')


# # 2. BERT fine-tuning

# Two fine-tuned models will be individual used for positive samples and negative.  
#   
# Hugging Face Transformers BERT models must be prepared in your Kaggle dataset for use in the off-line notebook.  
#   
# See also [bertforquestionanswering-base-uncased](https://www.kaggle.com/yutanakamura/bertforquestionanswering-base-uncased).

# In[ ]:


pos_model = BertForQuestionAnswering.from_pretrained('../input/bertforquestionanswering-base-uncased')
neg_model = BertForQuestionAnswering.from_pretrained('../input/bertforquestionanswering-base-uncased')


# In[ ]:


MAX_LENGTH = 128
BATCH_SIZE = 32


# In[ ]:


class TrainDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.texts = df['uncased_text'].values
        self.start_ids = df['start_position'].values
        self.end_ids = df['end_position'].values
        self.hash_index = df['textID'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        returns = {
            'text' : self.texts[idx],
            'start' : self.start_ids[idx],
            'end' : self.end_ids[idx],
            'idx' : idx
        }
        return returns


# In[ ]:


class TestDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.texts = df['uncased_text'].values
        self.hash_index = df['textID'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        returns = {
            'text' : self.texts[idx],
            'idx' : idx
        }
        return returns


# In[ ]:


ds_pos_train = TrainDataset(pos_train)
ds_neg_train = TrainDataset(neg_train)

ds_pos_val = TrainDataset(pos_val)
ds_neg_val = TrainDataset(neg_val)

ds_pos_test = TestDataset(pos_test)
ds_neg_test = TestDataset(neg_test)


# In[ ]:


dl_pos_train = DataLoader(ds_pos_train, batch_size=BATCH_SIZE, shuffle=True)
dl_neg_train = DataLoader(ds_neg_train, batch_size=BATCH_SIZE, shuffle=True)

dl_pos_val = DataLoader(ds_pos_val, batch_size=BATCH_SIZE, shuffle=False)
dl_neg_val = DataLoader(ds_neg_val, batch_size=BATCH_SIZE, shuffle=False)

dl_pos_test = DataLoader(ds_pos_test, batch_size=BATCH_SIZE, shuffle=False)
dl_neg_test = DataLoader(ds_neg_test, batch_size=BATCH_SIZE, shuffle=False)


# In[ ]:


class BaseSuperModule(pl.LightningModule):
    def __init__(self, bertmodel, tokenizer, prediction_save_path):
        super().__init__()
        self.bertmodel = bertmodel
        self.tokenizer = tokenizer
        self.prediction_save_path = prediction_save_path

    def get_device(self):
        return self.bertmodel.state_dict()['bert.embeddings.word_embeddings.weight'].device

    def save_predictions(self, start_positions, end_positions):
        d = pd.DataFrame({'start_position':start_positions, 'end_position':end_positions})
        d.to_csv(self.prediction_save_path, index=False)

    def forward(self, batch):
        """
        Input:
            batch(dict), where
                batch['text'] = uncased text: str
                batch['idx'] = raw text: list(int)
                batch['start'] = start position indices : list(int) (for train & val batch only)
                batch['end'] = end position indices : list(int) (for train & val batch only)

        Output:
            For train batch, which has 'start' key and 'end' key:
                Tuple of (loss(int), start_score(torch.tensor), end_score(torch.tensor))
            For test batch, without 'start' key and 'end' key:
                Tuple of (start_score(torch.tensor), end_score(torch.tensor))
        """
        encoded_batch = tokenizer.batch_encode_plus(batch['text'], max_length=MAX_LENGTH, pad_to_max_length=True)
        input_ids = torch.tensor(encoded_batch['input_ids']).to(self.get_device())
        attention_mask = torch.tensor(encoded_batch['attention_mask']).to(self.get_device())
        start_positions = batch['start'].to(self.get_device()) + 1  if 'start' in batch.keys() else None
        end_positions = batch['end'].to(self.get_device()) + 1  if 'end' in batch.keys() else None

        model_inputs = {
            'input_ids' : input_ids,
            'attention_mask' : attention_mask,
            'start_positions' : start_positions,
            'end_positions' : end_positions
        }
        
        return self.bertmodel(**model_inputs)

    def training_step(self, batch, batch_nb):
        """
        (batch) -> (dict or OrderedDict)
        # Caution: key for loss function must exactly be 'loss'.
        """
        idx = batch['idx']
        loss = self.forward(batch)[0]
        return {'loss':loss, 'idx':idx}

    def validation_step(self, batch, batch_nb):
        """
        (batch) -> (dict or OrderedDict)
        # Caution: key for loss function must exactly be 'loss'.
        """
        idx = batch['idx']
        loss = self.forward(batch)[0]
        return {'loss':loss, 'idx':idx}

    def test_step(self, batch, batch_nb):
        """
        (batch) -> (dict or OrderedDict)
        """
        idx = batch['idx']
        start_scores = self.forward(batch)[0]
        end_scores = self.forward(batch)[1]
        return {'start_scores':start_scores, 'end_scores':end_scores, 'idx':idx}

    def training_end(self, outputs):
        """
        outputs(dict) -> loss(dict or OrderedDict)
        # Caution: key must exactly be 'loss'.
        """
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
        self.save_predictions(start_positions, end_positions)
        return {}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=2e-5)

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


class PositiveModule(BaseSuperModule):
    def __init__(self, bertmodel, tokenizer, prediction_save_path):
        super().__init__(bertmodel, tokenizer, prediction_save_path)

    @pl.data_loader
    def train_dataloader(self):
        return dl_pos_train

    @pl.data_loader
    def val_dataloader(self):
        return dl_pos_val

    @pl.data_loader
    def test_dataloader(self):
        return dl_pos_test


# In[ ]:


class NegativeModule(BaseSuperModule):
    def __init__(self, bertmodel, tokenizer, prediction_save_path):
        super().__init__(bertmodel, tokenizer, prediction_save_path)

    @pl.data_loader
    def train_dataloader(self):
        return dl_neg_train

    @pl.data_loader
    def val_dataloader(self):
        return dl_neg_val

    @pl.data_loader
    def test_dataloader(self):
        return dl_neg_test


# In[ ]:


pos_module = PositiveModule(pos_model, tokenizer, 'pos_pred.csv')
neg_module = NegativeModule(neg_model, tokenizer, 'neg_pred.csv')


# In[ ]:


device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'


# In[ ]:


pos_module.to(device)
neg_module.to(device)


# In[ ]:


pos_trainer = pl.Trainer(max_nb_epochs=3, fast_dev_run=DEBUG_MODE)
neg_trainer = pl.Trainer(max_nb_epochs=3, fast_dev_run=DEBUG_MODE)


# In[ ]:


pos_trainer.fit(pos_module)
neg_trainer.fit(neg_module)


# In[ ]:


pos_trainer.test()
neg_trainer.test()


# # 3. Postprocessing

# The previous cell will save prediction results as pos_pred.csv and neg_pred.csv.  
# Let us load them.

# In[ ]:


get_ipython().system('ls')


# In[ ]:


pos_pred = pd.read_csv('pos_pred.csv')
neg_pred = pd.read_csv('neg_pred.csv')
pos_pred


# Apply predicsion results onto df_test for positive & negative samples.

# In[ ]:


df_test.index = df_test['textID']
df_test['selected_text'] = ''

df_test.loc[ds_pos_test.hash_index[:BATCH_SIZE if DEBUG_MODE else len(df_test)], 'start_position':'end_position'] = pos_pred.values
df_test.loc[ds_neg_test.hash_index[:BATCH_SIZE if DEBUG_MODE else len(df_test)], 'start_position':'end_position'] = neg_pred.values
df_test


# Convert start & end position into selected texts.

# In[ ]:


for i in tqdm(range(len(df_test))):
    if df_test['sentiment'].iloc[i] in ('positive', 'negative'):
        tokenized_text = df_test['tokenized_text'].iloc[i]
        start_position = max(df_test['start_position'].iloc[i], 0)
        end_position = min(df_test['end_position'].iloc[i], len(tokenized_text) - 1)
        
        # restore original text
        selected_text = tokenizer.convert_tokens_to_string(tokenized_text[start_position : end_position + 1])
        for original_token in df_test['text'].iloc[i].split():
            tokenized_form = tokenizer.convert_tokens_to_string(tokenizer.tokenize(original_token))
            selected_text = selected_text.replace(tokenized_form, original_token, 1)
        
        df_test['selected_text'].iloc[i] = selected_text


# For neutral samples, use original texts as they are.

# In[ ]:


for i in tqdm(range(len(df_test))):
    if df_test['sentiment'].iloc[i] == 'neutral':
        df_test['selected_text'].iloc[i] = df_test['text'].iloc[i]
    else:
        pass


# In[ ]:


df_test.loc[:, ['textID', 'selected_text']]


# In[ ]:


df_test.loc[:, ['textID', 'selected_text']].to_csv('submission.csv', index=False)


# Hope this notebook will help you all.  
# 
# Your upvotes and comments will greatly be appreciated.
# 
# Let us enjoy NLP!! ;)

# In[ ]:




