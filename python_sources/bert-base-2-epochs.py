#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install iterative-stratification')


# In[ ]:


import sys
package_dir = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"
sys.path.append(package_dir)


# In[ ]:


# standard imports
import time
import random
import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
# pytorch imports
from torch.optim import lr_scheduler
import torch
import shutil
import torch.nn as nn
import torch.utils.data
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer as keras_tokenizer
# cross validation and metrics
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
# progress bars
import nltk
import re
import collections

from tqdm import tqdm
tqdm.pandas()

from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertConfig


# In[ ]:


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything()


# In[ ]:


train_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")
percent_true = train_df.target.mean()
print("Percent true", percent_true)
test_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")
print('Train data dimension: ', train_df.shape)
print('Test data dimension: ', test_df.shape)


# In[ ]:


y_train = train_df['target'].values

identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

y_train_aux = train_df[['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values

def convert_to_bool(df, col_name):
    train_df[col_name] = np.where(train_df[col_name] >= 0.5, True, False)
    
for col in ['target'] + identity_columns:
        convert_to_bool(train_df, col)


# In[ ]:


toxicity_ann = np.sqrt(train_df.toxicity_annotator_count)
identity_ann = np.sqrt(train_df.identity_annotator_count)

identity_ann = identity_ann.clip_upper(np.percentile(identity_ann, 95))
toxicity_ann = toxicity_ann.clip_upper(np.percentile(toxicity_ann, 95))

weights = np.ones((len(train_df),)) / 4 #* toxicity_ann
# Subgroup
weights += train_df[identity_columns].sum(axis=1).astype(np.int) / 4 #* identity_ann
# Background Positive, Subgroup Negative
weights += ((train_df['target'].astype(np.int) +
   (~train_df[identity_columns]).sum(axis=1).astype(np.int) ) > 1 ).astype(np.int) / 4 #* (identity_ann + toxicity_ann) / 2
# Background Negative, Subgroup Positive
weights += ((~train_df['target']).astype(np.int) +
   (train_df[identity_columns].sum(axis=1).astype(np.int) ) > 1 ).astype(np.int) / 4 #* (identity_ann + toxicity_ann) / 2
loss_weight = 1.0 / weights.mean()

y_train = np.vstack([y_train, weights, toxicity_ann]).T


# In[ ]:


MAXLEN = 220
BATCH_SIZE = 16
N_EPOCHS = 2
FOLDS = 4
MODEL = "bert_uncased"

if not os.path.exists(MODEL):
    os.makedirs(MODEL)


# In[ ]:


# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# for df in [train_df, test_df]:
#     df["comment_text"] = df["comment_text"].progress_apply(lambda x: tokenizer.tokenize(x.lower()))
# for df in [train_df, test_df]:
#     df["tokens"] = df["comment_text"].progress_apply(lambda x:\
#                                         tokenizer.convert_tokens_to_ids(x[:MAXLEN]))
# x_train = train_df.logits.values   

x_train = np.load("../input/bert-train/train2.npy",  allow_pickle=True)


# In[ ]:


BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'
convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
    BERT_MODEL_PATH + 'bert_model.ckpt',
    BERT_MODEL_PATH + 'bert_config.json',
    MODEL + '/pytorch_model.bin')

shutil.copyfile(BERT_MODEL_PATH + 'bert_config.json', MODEL + '/bert_config.json')


# In[ ]:


def get_final_auc(indexes, preds):
    SUBGROUP_AUC = 'subgroup_auc'
    BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
    BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive

    def compute_auc(y_true, y_pred):
        try:
            return roc_auc_score(y_true >= 0.5, y_pred)
        except ValueError:
            return np.nan

    def compute_subgroup_auc(df, subgroup, label, model_name):
        subgroup_examples = df[df[subgroup]]
        return compute_auc(subgroup_examples[label], subgroup_examples[model_name])

    def compute_bpsn_auc(df, subgroup, label, model_name):
        """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
        subgroup_negative_examples = df[df[subgroup] & ~df[label]]
        non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
        examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
        return compute_auc(examples[label], examples[model_name])

    def compute_bnsp_auc(df, subgroup, label, model_name):
        """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
        subgroup_positive_examples = df[df[subgroup] & df[label]]
        non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
        examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
        return compute_auc(examples[label], examples[model_name])

    def compute_bias_metrics_for_model(dataset,
                                       subgroups,
                                       model,
                                       label_col,
                                       include_asegs=False):
        """Computes per-subgroup metrics for all subgroups and one model."""
        records = []
        for subgroup in subgroups:
            record = {
                'subgroup': subgroup,
                'subgroup_size': len(dataset[dataset[subgroup] >= 0.5])
            }
            record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
            record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
            record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
            records.append(record)
        return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)


    def convert_to_bool(df, col_name):
        df[col_name] = np.where(df[col_name] >= 0.5, True, False)

    for ic in ['target'] + identity_columns:
        convert_to_bool(train_df, ic)

    def calculate_overall_auc(df, model_name):
        true_labels = df["target"]
        predicted_labels = df[model_name]
        return roc_auc_score(true_labels, predicted_labels)

    def power_mean(series, p):
        total = sum(np.power(series, p))
        return np.power(total / len(series), 1 / p)

    def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
        bias_score = np.average([
            power_mean(bias_df[SUBGROUP_AUC], POWER),
            power_mean(bias_df[BPSN_AUC], POWER),
            power_mean(bias_df[BNSP_AUC], POWER)
        ])
        return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)
    
    train_df.loc[indexes, "predictions"] = preds
    bias_metrics_df = compute_bias_metrics_for_model(train_df.loc[indexes], identity_columns, "predictions", 'target')
    final_auc = get_final_metric(bias_metrics_df, calculate_overall_auc(train_df.loc[indexes], "predictions"))
    train_df.drop("predictions", axis=1, inplace=True)
    
    return final_auc


# In[ ]:


class NeuralNet(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(NeuralNet, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

def pad_batch(batch):
    sequences, targets, targets_aux = zip(*batch)
    max_length = max([len(s) for s in sequences])

    return pad_sequences(sequences, maxlen=min(max_length, MAXLEN)), targets, targets_aux
            
class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequences, targets, targets_aux, is_test = False):
        self.sequences = sequences
        self.lengths = [len(x) for x in sequences]
        self.targets = targets
        self.targets_aux = targets_aux
        self.is_test = is_test
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        if self.is_test:
            return (self.sequences[index], None, None)
        else:
            return self.sequences[index], self.targets[index], self.targets_aux[index]


# In[ ]:


def validate(model, valid_loader):
    model.eval()
    valid_preds_fold = np.zeros((len(valid_idx)))

    for i, (x_batch, y_batch, _) in enumerate(valid_loader):
        x_batch = torch.tensor(x_batch, dtype=torch.long).cuda()        
        y_batch = torch.tensor(y_batch, dtype=torch.float32).cuda()

        y_pred = model(x_batch, attention_mask=(x_batch > 0).cuda()).detach()
        
        valid_preds_fold[i * BATCH_SIZE:(i+1) * BATCH_SIZE] = y_pred.sigmoid().cpu().numpy()[:, 0]
    
    valid_preds_fold = valid_preds_fold[val_sampler.get_reverse_indexes()]
    elapsed_time = time.time() - start_time 
    roc_val = roc_auc_score(y_train[valid_idx, 0] > 0.5, valid_preds_fold)
    b_roc_val = get_final_auc(valid_idx, valid_preds_fold)
    
    print('Epoch {}/{} \t roc_auc={:.4f} \t b_roc_auc={:.4f} \t time={:.2f}s'.format(
        epoch + 1, N_EPOCHS, roc_val, b_roc_val, elapsed_time))
    
    return valid_preds_fold, b_roc_val


# In[ ]:


def save_checkpoint(checkpoint_path, model, optimizer):
    checkpoint_path = os.path.join(MODEL, checkpoint_path)
    state = {'state_dict': model.state_dict()}
    if optimizer:
        state['optimizer'] = optimizer.state_dict()
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint_path = os.path.join(MODEL, checkpoint_path)
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    if optimizer:
        optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)


# In[ ]:


#  bucket iterator
def divide_chunks(l, n):
    if n == len(l):
        yield np.arange(len(l), dtype=np.int32), l
    else:
        # looping till length l
        for i in range(0, len(l), n):
            data = l[i:i + n]
            yield np.arange(i, i + len(data), dtype=np.int32), data


def prepare_buckets(lens, bucket_size, batch_size, shuffle_data=True, indices=None):
    lens = -lens
    assert bucket_size % batch_size == 0 or bucket_size == len(lens)
    if indices is None:
        if shuffle_data:
            indices = shuffle(np.arange(len(lens), dtype=np.int32))
            lens = lens[indices]
        else:
            indices = np.arange(len(lens), dtype=np.int32)
    new_indices = []
    extra_batch = None
    for chunk_index, chunk in (divide_chunks(lens, bucket_size)):
        # sort indices in bucket by descending order of length
        indices_sorted = chunk_index[np.argsort(chunk, axis=-1)]
        batches = []
        for _, batch in divide_chunks(indices_sorted, batch_size):
            if len(batch) == batch_size:
                batches.append(batch.tolist())
            else:
                assert extra_batch is None
                assert batch is not None
                extra_batch = batch
        # shuffling batches within buckets
        if shuffle_data:
            batches = shuffle(batches)
        for batch in batches:
            new_indices.extend(batch)

    if extra_batch is not None:
        new_indices.extend(extra_batch)
    return indices[new_indices]


class BucketSampler(torch.utils.data.Sampler):

    def __init__(self, data_source, sort_keys, bucket_size=None, batch_size=1536, shuffle_data=True):
        super().__init__(data_source)
        self.shuffle = shuffle_data
        self.batch_size = batch_size
        self.sort_keys = sort_keys
        self.bucket_size = bucket_size if bucket_size is not None else len(sort_keys)
        if not shuffle_data:
            self.index = prepare_buckets(self.sort_keys, bucket_size=self.bucket_size, batch_size=self.batch_size,
                                         shuffle_data=self.shuffle)
        else:
            self.index = None
        self.weights = None

    def set_weights(self, w):
        assert w >= 0
        total = np.sum(w)
        if total != 1:
            w = w / total
        self.weights = w

    def __iter__(self):
        indices = None
        if self.weights is not None:
            total = len(self.sort_keys)
            
            indices = np.random.choice(total, (total,), p=self.weights)
        if self.shuffle:
            self.index = prepare_buckets(self.sort_keys, bucket_size=self.bucket_size, batch_size=self.batch_size,
                                         shuffle_data=self.shuffle, indices=indices)

        return iter(self.index)

    def get_reverse_indexes(self):
        indexes = np.zeros((len(self.index),), dtype=np.int32)
        for i, j in enumerate(self.index):
            indexes[j] = i
        return indexes

    def __len__(self):
        return len(self.sort_keys)


# In[ ]:


train_preds = np.zeros((len(train_df)))
splits = list(MultilabelStratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42).split(x_train, train_df[identity_columns + ["target"]]))


# In[ ]:


seed_everything()
FOLD = 0
for fold, (train_idx, valid_idx) in enumerate(splits):   
    if fold != FOLD:
        continue
    train = Dataset([x_train[i] for i in train_idx], 
                    y_train[train_idx],
                    y_train_aux[train_idx])
    valid = Dataset([x_train[i] for i in  valid_idx], 
                    y_train[valid_idx],
                    y_train_aux[valid_idx])
    
    model = NeuralNet.from_pretrained(MODEL,
            num_labels=6)
    model.cuda()
    
    loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, gamma=1/4, step_size=1)

    train_sampler = BucketSampler(train, 
                     np.array(train.lengths), 
                     bucket_size=BATCH_SIZE*1000,
                     batch_size=BATCH_SIZE,
                     shuffle_data=True)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, 
                sampler=train_sampler, collate_fn=pad_batch)
    
    val_sampler = BucketSampler(valid, 
                     np.array(valid.lengths), 
                     batch_size=BATCH_SIZE,
                     shuffle_data=False)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=BATCH_SIZE, 
                sampler=val_sampler, collate_fn=pad_batch)
    
    
    print('Fold ', fold)
    f1_best_best = 0.91
    
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        model.train()
        
        for itr, (x_batch, y_batch, y_batch_aux) in tqdm(enumerate(train_loader), disable=True):
            x_batch = torch.tensor(x_batch, dtype=torch.long).cuda()
            y_batch = torch.tensor(y_batch, dtype=torch.float32).cuda()
            y_batch_aux = torch.tensor(y_batch_aux, dtype=torch.float32).cuda()

            y_pred = model(x_batch, attention_mask=(x_batch > 0).cuda())
 
            loss = loss_weight * loss_fn(y_pred[:, 0], y_batch[:,0], y_batch[:,1])
            loss += loss_fn(y_pred[:, 1:], y_batch_aux)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

           
        valid_preds_fold, f1_best = validate(model, valid_loader)
        if f1_best > f1_best_best:
            f1_best_best = f1_best
            train_preds[valid_idx] = valid_preds_fold
            save_checkpoint("f{}.pth".format(fold), model, None)

        scheduler.step()


# In[ ]:




