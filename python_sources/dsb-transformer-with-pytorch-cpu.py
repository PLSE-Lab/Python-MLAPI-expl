#!/usr/bin/env python
# coding: utf-8

# ## TL;DR
# 
# ### This is inspired by https://www.kaggle.com/toshik/37th-place-solution/notebook

# In[ ]:


import copy
import gc
import json
import math
import os
import pickle
import random
import re
import six
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import lr_scheduler
from torch.utils.data import (Dataset,DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from pathlib import Path
from functools import partial, reduce
from collections import Counter, defaultdict
from contextlib import contextmanager
from torch.autograd import Variable

import multiprocessing
from multiprocessing import Process

from tqdm import tqdm_notebook as tqdm
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy as sp
from scipy import optimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, train_test_split
from numba import jit
from IPython.display import display



SEED = 1129

def seed_everything(seed=1129):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(SEED)


def to_pickle(filename, obj):
    with open(filename, mode='wb') as f:
        pickle.dump(obj, f)

def unpickle(filename):
    with open(filename, mode='rb') as fo:
        p = pickle.load(fo)
    return p  

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('[{}] done in {} s'.format(name, round(time.time() - t0, 2)))


# In[ ]:


warnings.filterwarnings('ignore')

dir_dataset = Path('/kaggle/input/data-science-bowl-2019')

dir_model = Path('/kaggle/input/dsb2019-37th-models')

input_path = "/kaggle/input/data-science-bowl-2019/"


# In[ ]:


def read_data():
    print('Reading test.csv file....')
    test = pd.read_csv(input_path + 'test.csv')
    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv(input_path + 'train_labels.csv')
    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv(input_path + 'specs.csv')
    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv(input_path + 'sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))
    return test, train_labels, specs, sample_submission


def read_pickle_data():
    print('Reading train.pkl file....')
    train = unpickle('../input/dsb2019-raw-data-pickled/train.pkl')
    print('Training.pkl file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))
    train_feature_gp = unpickle('../input/dsb2019-raw-data-pickled/train_feature_gp.pkl')
    train_history = unpickle('../input/dsb2019-raw-data-pickled/train_history.pkl') 
    train_current = unpickle('../input/dsb2019-raw-data-pickled/train_current.pkl') 
    return train, train_feature_gp, train_history, train_current


# The entire model receive query features and history features.
# Encoder layer extract high level features from history and they are concatenated with query features.
# Subsequently, output values are calculated throught FC layers.

# ## 4.3 Feature Extraction
# Since history data is given to the model directly, feature extraction is very simple.
# 
# As a history feature, the number of event code, title, types are counted for each game_session.
# Duration of geme_session is also calculated and concatenated with history features.
# 
# As a query feature, the number of `correct` and `incorrect` are counted respectively. 

# In[ ]:


session_merge = partial(pd.merge, on='game_session')

def extract_features(data, data_labels, event_codes, titles, types, num_history_step: int):

    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['correct'] = data['event_data'].map(lambda x: '"correct":true' in x)
    data['incorrect'] = data['event_data'].map(lambda x: '"correct":false' in x)

    data_gp = data.groupby(['installation_id', 'game_session'])
    data_time = data_gp['timestamp'].agg(min).reset_index()

    data_time_max = data_gp['timestamp'].agg(max).reset_index()[['game_session', 'timestamp']]
    data_time_max.columns = ['game_session', 'timestamp_end']

    data_level = data_gp['f_level'].agg(np.max).reset_index()[['game_session', 'f_level']]
    data_level = data_level.fillna(0.0)

    data_count = data_gp[['correct', 'incorrect']].agg(sum).reset_index()[['game_session', 'correct', 'incorrect']]

    data_code = pd.crosstab(data['game_session'], data['event_code']).astype(np.float32)
    data_title = pd.crosstab(data['game_session'], data['title']).astype(np.float32)
    data_type = pd.crosstab(data['game_session'], data['type']).astype(np.float32)

    data_title_str = data.drop_duplicates('game_session', keep='last').copy()[['game_session', 'title']]

    data_feature = reduce(
        session_merge,
        [data_time, data_code, data_title, data_type, data_time_max, data_title_str, data_count, data_level]
    )
    data_feature.index = data_feature['game_session']
    data_feature_gp = data_feature.groupby('installation_id')

    list_history = list()
    list_current = list()

    num_unique_geme_session = len(set(data_feature['game_session']))
    num_unique_id_and_game_session = len(set(zip(data_feature['installation_id'], data_feature['game_session'])))
    assert num_unique_geme_session == num_unique_id_and_game_session

    assessments = [
        'Mushroom Sorter (Assessment)',
        'Bird Measurer (Assessment)',
        'Cauldron Filler (Assessment)',
        'Cart Balancer (Assessment)',
        'Chest Sorter (Assessment)'
    ]

    for _, row in tqdm(data_labels.iterrows(), total=len(data_labels), miniters=100):

        same_id = data_feature_gp.get_group(row['installation_id'])

        target_timestamp = same_id.loc[row['game_session'], 'timestamp']

        same_id_before = same_id.loc[same_id['timestamp'] < target_timestamp].copy()
        same_id_before.sort_values('timestamp', inplace=True)

        same_id_before['duration'] = (same_id_before['timestamp_end'] - same_id_before['timestamp']).dt.total_seconds()
        same_id_before['duration'] = np.log1p(same_id_before['duration'])

        h_feature = same_id_before.iloc[-num_history_step:][event_codes + titles + types + ['duration']]
        h_feature = np.log1p(h_feature.values)

        c_feature = (same_id.loc[row['game_session']][assessments].values != 0).astype(np.int32)

        query_title = row['title']
        success_exp = np.sum(same_id_before.query('title==@query_title')['correct'])
        failure_exp = np.sum(same_id_before.query('title==@query_title')['incorrect'])

        c_feature = np.append(c_feature, np.log1p(success_exp))
        c_feature = np.append(c_feature, np.log1p(failure_exp))
        c_feature = np.append(c_feature, (success_exp + 1) / (success_exp + failure_exp + 2) - 0.5)
        c_feature = np.append(c_feature, (target_timestamp.hour - 12.0) / 10.0)

        if len(h_feature) < num_history_step:
            h_feature = np.pad(h_feature, ((num_history_step - len(h_feature), 0), (0, 0)),
                               mode='constant', constant_values=0)

        list_history.append(h_feature)
        list_current.append(c_feature)

    history = np.asarray(list_history)
    current = np.asarray(list_current)

    return data_feature_gp, history, current


# In[ ]:


# https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation

def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


# ### load data

# In[ ]:


with timer('load data'):
    seed = 3048
    fold_id = 0
    epochs = 3
    batch_size = 64

    with open(str(dir_model / f'parameters_seed{seed}.json'), 'r') as f:
                hyper_params = json.load(f)

    num_history_step = hyper_params['num_history_step']
    
    # read data
    test, train_labels, specs, sample_submission = read_data()
    train, train_feature_gp, train_history, train_current = read_pickle_data()
    y = train_labels["accuracy_group"].values

    num_folds = 5
    # kf = StratifiedGroupKFold(n_splits = num_folds)
    # splits = list(kf.split(X=train_labels, y=y, groups=train_labels.installation_id))
    
    splits = stratified_group_k_fold(train_labels, y, train_labels.installation_id, num_folds, seed=SEED)
    # train_idx = splits[fold_id][0]
    # val_idx = splits[fold_id][1]
    # train_idx, val_idx = train_test_split(train_labels.index.tolist(), test_size=0.2, 
    #                                       random_state=SEED, stratify=y)
    
    gc.collect()


# ### train data feature engineering ###
# ### you can skip here with read_pickle_data() function

# In[ ]:


# train_installation_id = list(set(train_labels["installation_id"]))
# len(train_installation_id)

# train = train[train.installation_id.isin(train_installation_id)]
# print('train shape: {}'.format(train.shape))

# train.sort_values(['installation_id', 'timestamp'], inplace=True)
# print('train shape: {}'.format(train.shape))

event_codes = pd.read_csv(dir_model / f'event_codes.csv')['event_code'].tolist()
titles = pd.read_csv(dir_model / 'media_sequence.csv')['title'].tolist()
types = ['Activity', 'Assessment', 'Clip', 'Game']
# re_level = re.compile(r'.*"level":([0-9]+).*')

# train['event_code'] = pd.Categorical(train['event_code'], categories=event_codes)
# train['title'] = pd.Categorical(train['title'], categories=titles)
# train['type'] = pd.Categorical(train['type'], categories=types)
# train['f_level'] = train['event_data'].map(
#         lambda x: int(re.sub(re_level, '\\1', x)) + 1 if '"level"' in x else np.nan)

# print(' train shape: {}'.format(train.shape))

# train_feature_gp, train_history, train_current = extract_features(train, train_labels, event_codes, titles, types, num_history_step)

# train
# result_path = "train_history.pkl"
# to_pickle(result_path, train_history)

# result_path = "train_current.pkl"
# to_pickle(result_path, train_current)

# result_path = "train_feature_gp.pkl"
# to_pickle(result_path, train_feature_gp)


# ### Prepare Transformer model

# In[ ]:


class ConvolutionSentence(nn.Conv2d):
    """ Position-wise Linear Layer for Sentence Block
    Position-wise linear layer for array of shape
    (batchsize, dimension, sentence_length)
    can be implemented a convolution layer.
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ConvolutionSentence, self).__init__(
             in_channels, out_channels,
             kernel_size, stride, padding, dilation, groups, bias)

    def __call__(self, x):
        """Applies the linear layer.
        Args:
            x (~chainer.Variable): Batch of input vector block. Its shape is
                (batchsize, in_channels, sentence_length).
        Returns:
            ~chainer.Variable: Output of the linear layer. Its shape is
                (batchsize, out_channels, sentence_length).
        """     
        x = x.unsqueeze(3)
        y = super(ConvolutionSentence, self).__call__(x)
        y = torch.squeeze(y, 3)
        return y


# In[ ]:


class MultiHeadAttention(nn.Module):
    """ Multi Head Attention Layer for Sentence Blocks
    For batch computation efficiency, dot product to calculate query-key
    scores is performed all heads together.
    """

    def __init__(self, n_units, h=8, dropout=0.1, self_attention=True):
        super(MultiHeadAttention, self).__init__()
        
        if self_attention:
            self.W_QKV = ConvolutionSentence(
                n_units, n_units * 3, kernel_size=1, bias=False)
        else:
            self.W_Q = ConvolutionSentence(
                n_units, n_units, kernel_size=1, bias=False)
            self.W_KV = ConvolutionSentence(
                n_units, n_units * 2, kernel_size=1, bias=False)
        self.finishing_linear_layer = ConvolutionSentence(
            n_units, n_units, bias=False)
        self.h = h
        self.scale_score = 1. / (n_units // h) ** 0.5
        self.dropout = dropout
        self.is_self_attention = self_attention

    def __call__(self, x, z=None, mask=None):
        # xp = self.xp
        h = self.h

        # temporary mask
        mask = np.zeros((8, x.shape[2], x.shape[2]), dtype=np.bool)

        if self.is_self_attention:
            # print(f'self.W_QKV(x) shape : {self.W_QKV(x).shape}')
            Q, K, V = torch.chunk(self.W_QKV(x), 3, axis=1)
        else:
            Q = self.W_Q(x)
            K, V = torch.chunk(self.W_KV(z), 2, axis=1)
        batch, n_units, n_querys = Q.shape
        _, _, n_keys = K.shape

        # Calculate Attention Scores with Mask for Zero-padded Areas
        # Perform Multi-head Attention using pseudo batching
        # all together at once for efficiency
        
        batch_Q = torch.cat(torch.chunk(Q, h, axis=1), axis=0)
        batch_K = torch.cat(torch.chunk(K, h, axis=1), axis=0)
        batch_V = torch.cat(torch.chunk(V, h, axis=1), axis=0)
        assert(batch_Q.shape == (batch * h, n_units // h, n_querys))
        assert(batch_K.shape == (batch * h, n_units // h, n_keys))
        assert(batch_V.shape == (batch * h, n_units // h, n_keys))

        # print(f'batch_Q shape : {batch_Q.shape}')
        # print(f'batch_K shape : {batch_K.shape}')
        # print(f'batch_V shape : {batch_V.shape}')
        # mask = xp.concatenate([mask] * h, axis=0)
        batch_A = torch.matmul(batch_Q.permute(0, 2, 1), batch_K) * self.scale_score
        # print(f'batch_A shape : {batch_A.shape}')
        # Calculate Weighted Sum
        batch_A = batch_A.unsqueeze(1)
        batch_V = batch_V.unsqueeze(2)
        batch_C = torch.sum(batch_A * batch_V, axis=3)
        assert(batch_C.shape == (batch * h, n_units // h, n_querys))
        
        # print(f'batch_C shape : {batch_C.shape}')
        
        C = torch.cat(torch.chunk(batch_C, h, axis=0), axis=1)
        assert(C.shape == (batch, n_units, n_querys))
        C = self.finishing_linear_layer(C)
        # print(f'C shape : {C.shape}')
        return C


# In[ ]:


class FeedForwardLayer(nn.Module):
    def __init__(self, n_units: int, ff_inner: int, ff_slope: float):
        super(FeedForwardLayer, self).__init__()
        n_inner_units = n_units * ff_inner
        self.slope = ff_slope
        self.W_1 = ConvolutionSentence(n_units, n_inner_units)
        self.W_2 = ConvolutionSentence(n_inner_units, n_units)
        self.act = F.leaky_relu

    def __call__(self, e):
        e = self.W_1(e)
        e = self.act(e, negative_slope=self.slope)
        e = self.W_2(e)
        return e
    

def seq_func(func, x, reconstruct_shape=True):
    """ Change implicitly function's target to ndim=3
    Apply a given function for array of ndim 3,
    shape (batchsize, dimension, sentence_length),
    instead for array of ndim 2.
    """

    batch, units, length = x.shape
    e = x.permute(0, 2, 1).reshape(batch * length, units)
    e = func(e)
    if not reconstruct_shape:
        return e
    out_units = e.shape[1]
    e = e.reshape((batch, length, out_units)).permute(0, 2, 1)
    assert(e.shape == (batch, out_units, length))
    return e


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class LayerNormalizationSentence(LayerNorm):
    """ Position-wise Linear Layer for Sentence Block
    Position-wise layer-normalization layer for array of shape
    (batchsize, dimension, sentence_length).
    """

    def __init__(self, *args, **kwargs):
        super(LayerNormalizationSentence, self).__init__(*args, **kwargs)

    def __call__(self, x):
        y = seq_func(super(LayerNormalizationSentence, self).__call__, x)
        return y


# In[ ]:


class EncoderLayer(nn.Module):
    def __init__(self, n_units, ff_inner: int, ff_slope: float, h: int, dropout1: float, dropout2: float):
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(n_units, h)
        self.feed_forward = FeedForwardLayer(n_units, ff_inner, ff_slope)
        self.ln_1 = LayerNormalizationSentence(n_units, eps=1e-6)
        self.ln_2 = LayerNormalizationSentence(n_units, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)

    def __call__(self, e, xx_mask):
        sub = self.self_attention(e, e, xx_mask)
        e = e + self.dropout1(sub)
        e = self.ln_1(e)

        sub = self.feed_forward(e)
        e = e + self.dropout2(sub)
        e = self.ln_2(e)
        return e


# In[ ]:


class DSB2019Net(nn.Module):

    def __init__(self, dim_input: int, dim_enc: int, dim_fc: int,
                 ff_inner: int, ff_slope: float, head: int,
                 dropout1: float, dropout2: float, dropout3: float, **kwargs):
        '''
        from https://www.kaggle.com/toshik/37th-place-solution/notebook
        thank you for @toshi_k 's nice solution and imprements
        '''
        super(DSB2019Net, self).__init__()

        self.dropout3 = nn.Dropout(dropout3)

        self.cur_fc1 = nn.Linear(9, 128)
        self.cur_fc2 = nn.Linear(128, dim_input)

        self.hist_conv1 = ConvolutionSentence(dim_input, int(dim_enc))
        self.hist_enc1 = EncoderLayer(int(dim_enc), ff_inner, ff_slope, head, dropout1, dropout2)

        self.fc1 = nn.Linear(283, dim_fc)
        self.fc2 = nn.Linear(dim_fc, 1)

    def __call__(self, query, history, targets):

        out = self.predict(query, history)
        return out

    def predict(self, query, history, **kwargs):
        """
            query: [batch_size, feature]
            history: [batch_size, time_step, feature]
        """

        h_cur = F.leaky_relu(self.cur_fc1(query))
        h_cur = self.cur_fc2(h_cur)

        h_hist = history.permute(0, 2, 1)

        h_hist = self.hist_conv1(h_hist)
        
        # print(h_hist.shape, h_cur.shape)

        h_hist = self.hist_enc1(h_hist, xx_mask=None)
    
        h_hist_ave = torch.mean(h_hist, axis=2)
        h_hist_max, _ = torch.max(h_hist, axis=2)

        h = torch.cat([h_cur, h_hist_ave, h_hist_max], axis=1)
        # print(f'h shape : {h.shape}')
        
        h = self.dropout3(F.leaky_relu(self.fc1(h)))
        out = self.fc2(h)
        return out


# ### Training part

# In[ ]:


class DictDataset(object):

    """Dataset of a dictionary of datasets.
    It combines multiple datasets into one dataset. Each example is represented
    by a dictionary mapping a key to an example of the corresponding dataset.
    Args:
        datasets: Underlying datasets. The keys are used as the keys of each
            example. All datasets must have the same length.
    """

    def __init__(self, **datasets):
        if not datasets:
            raise ValueError('no datasets are given')
        length = None
        for key, dataset in six.iteritems(datasets):
            if length is None:
                length = len(dataset)
            elif length != len(dataset):
                raise ValueError(
                    'dataset length conflicts at "{}"'.format(key))
        self._datasets = datasets
        self._length = length

    def __getitem__(self, index):
        batches = {key: dataset[index]
                   for key, dataset in six.iteritems(self._datasets)}
        if isinstance(index, slice):
            length = len(six.next(six.itervalues(batches)))
            return [{key: batch[i] for key, batch in six.iteritems(batches)}
                    for i in six.moves.range(length)]
        else:
            return batches

    def __len__(self):
        return self._length


# In[ ]:


def train_one_epoch(model, train_loader, criterion, optimizer, device, steps_upd_logging=500, accumulation_steps=1):
    '''
    from : https://github.com/okotaku/kaggle_rsna2019_3rd_solution/blob/master/src/trainer.py
    '''
    model.train()

    total_loss = 0.0
    for step, (input_dic) in tqdm(enumerate(train_loader), total=len(train_loader)):
        for k in input_dic.keys():
            input_dic[k] = input_dic[k].to(device)
            
        history = input_dic['history']
        query = input_dic['query']
        targets = input_dic['targets']
        
        optimizer.zero_grad()

        logits = model(query, history, targets)

        loss = criterion(logits, targets)
        loss.backward()

        if (step + 1) % accumulation_steps == 0:  # Wait for several backward steps
            optimizer.step()  # Now we can do an optimizer step

        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            print('Train loss on step {} was {}'.format(step + 1, round(total_loss / (step + 1), 5)))


    return total_loss / (step + 1)


def validate(model, val_loader, criterion, device):
    '''
    from : https://github.com/okotaku/kaggle_rsna2019_3rd_solution/blob/master/src/trainer.py
    '''
    model.eval()

    val_loss = 0.0
    true_ans_list = []
    preds_cat = []
    for step, (input_dic) in tqdm(enumerate(val_loader), total=len(val_loader)):
        for k in input_dic.keys():
            input_dic[k] = input_dic[k].to(device)
            
        history = input_dic['history']
        query = input_dic['query']
        targets = input_dic['targets']

        logits = model(query, history, targets)

        loss = criterion(logits, targets)
        val_loss += loss.item()

        targets = targets.float().cpu().detach().numpy()
        logits = logits.float().cpu().detach().numpy().astype("float32")
        true_ans_list.append(targets)
        preds_cat.append(logits)

        del input_dic, targets, logits
        gc.collect()

    all_true_ans = np.concatenate(true_ans_list, axis=0)
    all_preds = np.concatenate(preds_cat, axis=0)

    return all_preds, all_true_ans, val_loss / (step + 1)


# In[ ]:


# from https://gist.github.com/jamesr2323/33c67ba5ac29880171b63d2c7f1acdc5
# Thanks https://discuss.pytorch.org/t/rmse-loss-function/16540

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


# In[ ]:


for fold_id, (train_idx, val_idx) in enumerate(splits):
    
    print(f'fold {fold_id} : start !')
    
    with timer('prepare validation data'):
        y_train = y[train_idx]
        train_dataset = DictDataset(history=train_history.astype(np.float32)[train_idx],
                                    query=train_current.astype(np.float32)[train_idx],
                                    targets=np.asarray(y_train, dtype=np.float32))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size*8, shuffle=True, num_workers=0, pin_memory=True)

        y_val = y[val_idx]
        val_dataset = DictDataset(history=train_history.astype(np.float32)[val_idx],
                                  query=train_current.astype(np.float32)[val_idx],
                                  targets=np.asarray(y_val, dtype=np.float32))

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size*4, shuffle=False, num_workers=0, pin_memory=True)

        del train_dataset, val_dataset
        gc.collect()
        
        
    with timer('create model'):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = DSB2019Net(len(event_codes + titles + types) + 1, **hyper_params)
        model = model.to(device)

        criterion = RMSELoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10)
        
        
    with timer('training loop'):
        EXP_ID = 'DSB transformer approach with pytorch'
        OUT_DIR = '/kaggle/working'
        best_score = 999
        best_epoch = 0
        for epoch in range(1, epochs + 1):

            print("Starting {} epoch...".format(epoch))
            tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            print('Mean train loss: {}'.format(round(tr_loss, 5)))

            val_pred, y_true, val_loss = validate(model, val_loader, criterion, device)
            score = np.sqrt(mean_squared_error(y_true, val_pred))
            print('Mean valid loss: {} score: {}'.format(round(val_loss, 5), round(score, 5)))
            if score < best_score:
                best_score = score
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(OUT_DIR, '{}_fold{}.pth'.format(EXP_ID, fold_id)))
                np.save(os.path.join(OUT_DIR, "{}_fold{}_val.npy".format(EXP_ID, fold_id)), val_pred)
                np.save(os.path.join(OUT_DIR, "{}_fold{}_true.npy".format(EXP_ID, fold_id)), y_true)
            scheduler.step()

        print("best score={} on epoch={}".format(best_score, best_epoch))


# In[ ]:


true_pred_path = [
    'DSB transformer approach with pytorch_fold0_true.npy',
    'DSB transformer approach with pytorch_fold1_true.npy',
    'DSB transformer approach with pytorch_fold2_true.npy',
    'DSB transformer approach with pytorch_fold3_true.npy',
    'DSB transformer approach with pytorch_fold4_true.npy',
]

true_pred_list = []
for p in true_pred_path:
    true_pred_list.append(np.load(p))
    
true_preds = np.concatenate(true_pred_list)
true_preds.shape


# In[ ]:


val_pred_path = [
    'DSB transformer approach with pytorch_fold0_val.npy',
    'DSB transformer approach with pytorch_fold1_val.npy',
    'DSB transformer approach with pytorch_fold2_val.npy',
    'DSB transformer approach with pytorch_fold3_val.npy',
    'DSB transformer approach with pytorch_fold4_val.npy',
]

val_pred_list = []
for p in val_pred_path:
    val_pred_list.append(np.load(p))
    
val_preds = np.concatenate(val_pred_list)
val_preds.shape


# ### Predict part

# In[ ]:


with timer('make test features'):
    tic = time.time()

    sub = pd.read_csv(dir_dataset / 'sample_submission.csv')

    test_installation_id = list(set(sub.installation_id))

    print('test installation id: {}'.format(test_installation_id[:10]))

    test = pd.read_csv(dir_dataset / 'test.csv')
    test = test[test.installation_id.isin(test_installation_id)]
    print('test shape: {}'.format(test.shape))

    test.sort_values(['installation_id', 'timestamp'], inplace=True)
    test_labels = test.drop_duplicates('installation_id', keep='last').copy()
    test_labels.reset_index(drop=True, inplace=True)
    test_labels['accuracy_group'] = -1  # dummy label

    event_codes = pd.read_csv(dir_model / f'event_codes.csv')['event_code'].tolist()
    titles = pd.read_csv(dir_model / 'media_sequence.csv')['title'].tolist()
    types = ['Activity', 'Assessment', 'Clip', 'Game']
    re_level = re.compile(r'.*"level":([0-9]+).*')

    test['event_code'] = pd.Categorical(test['event_code'], categories=event_codes)
    test['title'] = pd.Categorical(test['title'], categories=titles)
    test['type'] = pd.Categorical(test['type'], categories=types)
    test['f_level'] = test['event_data'].map(
        lambda x: int(re.sub(re_level, '\\1', x)) + 1 if '"level"' in x else np.nan)

    print(' test shape: {}'.format(test.shape))

    data_feature_gp, test_history, test_current = extract_features(test, test_labels, event_codes, titles, types, num_history_step)

    print(test_history.shape, test_current.shape)


# In[ ]:


test_dataset = DictDataset(history=test_history.astype(np.float32),
                           query=test_current.astype(np.float32),
                           targets=np.asarray(test_labels[['accuracy_group']], dtype=np.float32))

batch_size = 32
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size*6, shuffle=False, num_workers=0, pin_memory=True)


# In[ ]:


with timer('create test model'):
    
    n_units = hyper_params['dim_enc']
    dim_input = len(event_codes + titles + types) + 1
        
    model_path = [
        'DSB transformer approach with pytorch_fold0.pth',
        'DSB transformer approach with pytorch_fold1.pth',
        'DSB transformer approach with pytorch_fold2.pth',
        'DSB transformer approach with pytorch_fold3.pth',
        'DSB transformer approach with pytorch_fold4.pth',
    ]
    
    models = []
    for p in model_path:
        model = DSB2019Net(dim_input, **hyper_params)
        model.load_state_dict(torch.load(p))
        model.to(device)
        model.eval()
        models.append(model)


# In[ ]:


def predict(models, test_loader, device):
    '''
    from : https://github.com/okotaku/kaggle_rsna2019_3rd_solution/blob/master/src/trainer.py
    '''
    preds_cat = []
    for step, (input_dic) in tqdm(enumerate(test_loader), total=len(test_loader)):
        for k in input_dic.keys():
            input_dic[k] = input_dic[k].to(device)
            
        history = input_dic['history']
        query = input_dic['query']
        targets = input_dic['targets']
    
        logits = []
        for m in models:
            logits_ = m(query, history, targets)
            logits_ = logits_.float().cpu().detach().numpy().astype("float32")
            logits.append(logits_)
        preds_cat.append(np.mean(logits, axis=0))

        del input_dic, logits
        gc.collect()

    all_preds = np.concatenate(preds_cat, axis=0)

    return all_preds


# In[ ]:


with timer('nn predict'):
    all_preds = predict(models, test_loader, device)
    print(all_preds.shape)


# In[ ]:


sub.accuracy_group = np.round(all_preds).astype('int')
print(sub.accuracy_group.value_counts())
sub.head()


# In[ ]:


sub.to_csv('submission.csv', index=None)


# In[ ]:




