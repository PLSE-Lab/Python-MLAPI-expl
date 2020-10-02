#!/usr/bin/env python
# coding: utf-8

# Thanks for @christofhenkel @abhishek @iezepov for their great work:
# 
# https://www.kaggle.com/christofhenkel/how-to-preprocessing-for-glove-part2-usage
# https://www.kaggle.com/abhishek/pytorch-bert-inference
# https://www.kaggle.com/iezepov/starter-gensim-word-embeddings

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import fastai
from fastai.train import Learner
from fastai.train import DataBunch
from fastai.callbacks import GeneralScheduler, TrainingPhase
from fastai.basic_data import DatasetType
import fastprogress
from fastprogress import force_console_behavior

import numpy as np
from pprint import pprint
import pandas as pd
import os
import time
import gc
import random
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
from keras.preprocessing import text, sequence
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F

import torch.utils.data
from tqdm import tqdm
import warnings
from nltk.tokenize.treebank import TreebankWordTokenizer
from scipy.stats import rankdata

from gensim.models import KeyedVectors

import copy


# In[ ]:


def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in tqdm(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    return np.array(all_tokens)

def is_interactive():
    return 'SHLVL' not in os.environ

def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    #with open(path,'rb') as f:
    emb_arr = KeyedVectors.load(path)
    return emb_arr

def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((max_features + 1, 300))
    unknown_words = []
    
    for word, i in word_index.items():
        if i <= max_features:
            try:
                embedding_matrix[i] = embedding_index[word]
            except KeyError:
                try:
                    embedding_matrix[i] = embedding_index[word.lower()]
                except KeyError:
                    try:
                        embedding_matrix[i] = embedding_index[word.title()]
                    except KeyError:
                        unknown_words.append(word)
    return embedding_matrix, unknown_words

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

def train_model(learn,output_dim,lr=0.001,
                batch_size=512, n_epochs=5):
    
    n = len(learn.data.train_dl)
    phases = [(TrainingPhase(n).schedule_hp('lr', lr * (0.6**(i)))) for i in range(n_epochs)]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)
    for epoch in range(n_epochs):
        learn.fit(1)
        learn.save('model_1_{}'.format(epoch))
        
    return learn

def handle_punctuation(x):
    x = x.translate(remove_dict)
    x = x.translate(isolate_dict)
    return x

def handle_contractions(x):
    x = tokenizer.tokenize(x)
    return x

def fix_quote(x):
    x = [x_[1:] if x_.startswith("'") else x_ for x_ in x]
    x = ' '.join(x)
    return x

def preprocess(x):
    x = handle_punctuation(x)
    x = handle_contractions(x)
    x = fix_quote(x)
    return x

class SequenceBucketCollator():
    def __init__(self, choose_length, sequence_index, length_index, label_index=None):
        self.choose_length = choose_length
        self.sequence_index = sequence_index
        self.length_index = length_index
        self.label_index = label_index
        
    def __call__(self, batch):
        batch = [torch.stack(x) for x in list(zip(*batch))]
        
        sequences = batch[self.sequence_index]
        lengths = batch[self.length_index]
        
        length = self.choose_length(lengths)
        mask = torch.arange(start=maxlen, end=0, step=-1) < length
        padded_sequences = sequences[:, mask]
        
        batch[self.sequence_index] = padded_sequences
        
        if self.label_index is not None:
            return [x for i, x in enumerate(batch) if i != self.label_index], batch[self.label_index]
    
        return batch

    
class SoftmaxPooling(nn.Module):
    def __init__(self, dim=1):
        super(self.__class__, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return (x * x.softmax(dim=self.dim)).sum(dim=self.dim)


# class AttentivePooling(nn.Module):
#     def __init__(self, dim=-1, input_size=128, hidden_size=32):
#         super(self.__class__, self).__init__()
#         self.dim = dim
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.nn_attn = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, 1)
#         )
        
#     def forward(self, x):
#         return (x * torch.transpose(self.nn_attn(torch.transpose(x, 2, 1)), 1, 2).softmax(dim=self.dim)).sum(dim=self.dim)


class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix, num_aux_targets):
        super(NeuralNet, self).__init__()
        embed_size = embedding_matrix.shape[1]
        
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
    
#         self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
#         self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        
        self.linear_out = nn.Sequential(
            nn.Dropout(0.5),
            nn.BatchNorm1d(DENSE_HIDDEN_UNITS),
            nn.Linear(DENSE_HIDDEN_UNITS, 1)
        )
        
        self.linear_aux_out = nn.Sequential(
            nn.Dropout(0.5),
            nn.BatchNorm1d(DENSE_HIDDEN_UNITS),
            nn.Linear(DENSE_HIDDEN_UNITS, num_aux_targets)
        )
        
        self.softmaxpool = SoftmaxPooling()
#         self.attnpool = AttentivePooling()
        
#         self.conv_extractor1 = nn.Sequential(
#             nn.Dropout(0.1),
#             nn.Conv1d(embed_size, 64, kernel_size=3, padding=1),
#             nn.BatchNorm1d(64),
#             nn.ELU()
#         )
        
#         self.conv_extractor2 = nn.Sequential(
#             nn.Dropout(0.1),
#             nn.Conv1d(embed_size, 64, kernel_size=5, padding=2),
#             nn.BatchNorm1d(64),
#             nn.ELU()
#         )
        
    def forward(self, x, lengths=None):
        h_embedding = self.embedding(x.long())
        h_embedding = self.embedding_dropout(h_embedding)
        
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)
        soft_pool = self.softmaxpool(h_lstm2)
        # attn_pool = self.attnpool(h_lstm2)
        
#         conv_feat1 = torch.transpose(self.conv_extractor1(torch.transpose(h_lstm1, 2, 1)), 1, 2)
        
        h_conc = torch.cat((max_pool, avg_pool, soft_pool), 1)
#         h_conc_linear1  = F.relu(self.linear1(h_conc))
#         except:
#             print(h_lstm2.size())
#             print(soft_pool.size())
#             print(h_conc.size())
#             raise RuntimeError('hello there')
#         h_conc_linear2  = F.relu(self.linear2(h_conc))
        
        hidden = h_conc
        
        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)
        
        return out
    
def custom_loss(data, targets):
    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:,1:2])(data[:,:1],targets[:,:1])
    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:,1:],targets[:,2:])
    return (bce_loss_1 * loss_weight) + bce_loss_2

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def ensemble_predictions(predictions, weights, type_="linear"):
    assert np.isclose(np.sum(weights), 1.0)
    if type_ == "linear":
        res = np.average(predictions, weights=weights, axis=0)
    elif type_ == "harmonic":
        res = np.average([1 / p for p in predictions], weights=weights, axis=0)
        return 1 / res
    elif type_ == "geometric":
        numerator = np.average(
            [np.log(p) for p in predictions], weights=weights, axis=0
        )
        res = np.exp(numerator / sum(weights))
        return res
    elif type_ == "rank":
        res = np.average([rankdata(p) for p in predictions], weights=weights, axis=0)
        return res / (len(res) + 1)
    return res


# In[ ]:


warnings.filterwarnings(action='once')
device = torch.device('cuda')
SEED = 1234
BATCH_SIZE = 512
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

tqdm.pandas()
CRAWL_EMBEDDING_PATH = '../input/gensim-embeddings-dataset/crawl-300d-2M.gensim'
PARAGRAM_EMBEDDING_PATH = '../input/gensim-embeddings-dataset/paragram_300_sl999.gensim'
NUM_MODELS = 1
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 768
if not is_interactive():
    def nop(it, *a, **k):
        return it

    tqdm = nop

    fastprogress.fastprogress.NO_BAR = True
    master_bar, progress_bar = force_console_behavior()
    fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar

seed_everything()


# **LSTM Part**

# In[ ]:


x_train = pd.read_csv('../input/jigsawbiaspreprocessed/x_train.csv', header=None)[0].astype('str')
y_aux_train = np.load('../input/jigsawbiaspreprocessed/y_aux_train.npy')
y_train = np.load('../input/jigsawbiaspreprocessed/y_train.npy')

loss_weight = 3.209226860170181

max_features = 400000


# In[ ]:


train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
annot_idx = train[train['identity_annotator_count'] > 0].sample(n=48660, random_state=13).index
not_annot_idx = train[train['identity_annotator_count'] == 0].sample(n=48660, random_state=13).index
x_val_idx = list(set(annot_idx).union(set(not_annot_idx)))
x_train_idx = list(set(x_train.index) - set(x_val_idx))


# In[ ]:


X_train = x_train.loc[x_train_idx]
Y_train = y_train[x_train_idx]
Y_aux_train = y_aux_train[x_train_idx]
X_val = x_train.loc[x_val_idx]
Y_val = y_train[x_val_idx]
Y_aux_val = y_aux_train[x_val_idx]


# In[ ]:


tokenizer = text.Tokenizer(num_words = max_features, filters='',lower=False)


# In[ ]:


tokenizer.fit_on_texts(list(X_train))

crawl_matrix, unknown_words_crawl = build_matrix(tokenizer.word_index, CRAWL_EMBEDDING_PATH)
print('n unknown words (crawl): ', len(unknown_words_crawl))

paragram_matrix, unknown_words_paragram = build_matrix(tokenizer.word_index, PARAGRAM_EMBEDDING_PATH)
print('n unknown words (paragram): ', len(unknown_words_paragram))

max_features = max_features or len(tokenizer.word_index) + 1
max_features

embedding_matrix = np.concatenate([crawl_matrix, paragram_matrix], axis=-1)
print(embedding_matrix.shape)

del crawl_matrix
del paragram_matrix
gc.collect()

y_train_torch = torch.tensor(np.hstack([Y_train, Y_aux_train]), dtype=torch.float32)
X_train = tokenizer.texts_to_sequences(X_train)


# In[ ]:


# https://stackoverflow.com/questions/45735070/keras-text-preprocessing-saving-tokenizer-object-to-file-for-scoring
import pickle

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


lengths = torch.from_numpy(np.array([len(x) for x in X_train]))
 
maxlen = 300
X_train_padded = torch.from_numpy(sequence.pad_sequences(X_train, maxlen=maxlen))


# In[ ]:


batch_size = 512
train_dataset = data.TensorDataset(X_train_padded, lengths, y_train_torch)
valid_dataset = data.Subset(train_dataset, indices=[0, 1])

train_collator = SequenceBucketCollator(lambda lenghts: lenghts.max(), 
                                        sequence_index=0, 
                                        length_index=1, 
                                        label_index=2)

train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collator)
valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=train_collator)

databunch = DataBunch(train_dl=train_loader, valid_dl=valid_loader, collate_fn=train_collator)


# In[ ]:


for model_idx in range(NUM_MODELS):
    print('Model ', model_idx)
    seed_everything(1 + model_idx)
    model = NeuralNet(embedding_matrix, Y_aux_train.shape[-1])
    learn = Learner(databunch, model, loss_func=custom_loss)
    train_model(learn,output_dim=7)

