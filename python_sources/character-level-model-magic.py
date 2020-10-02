#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This notebook shows of how we trained our 2nd level models, which are the key of our solution.
# 
# It assumes that predictions were already generated using transformers, and converted to character level with the following function :

# In[ ]:


def token_level_to_char_level(text, offsets, preds):
    probas_char = np.zeros(len(text))
    for i, offset in enumerate(offsets):
        if offset[0] or offset[1]: # remove padding and sentiment
            probas_char[offset[0]:offset[1]] = preds[i]
    
    return probas_char


# Those probabilities for all our models are available [here](https://www.kaggle.com/theoviel/tweet-char-lvl-preds).
# 
# **Enjoy !**

# ## Initialization

# In[ ]:


import re
import os
import gc
import time
import torch
import pickle
import string
import random
import warnings
import datetime
import itertools
import tokenizers
import numpy as np
import transformers
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import *
from torch.nn import functional as F


# from torchcontrib.optim import SWA
from torch.utils.data.sampler import *
from torch.utils.data import DataLoader
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.sequence import pad_sequences
from transformers import AdamW, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")


# In[ ]:


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
SEED = 2020
seed_everything(SEED)


# In[ ]:


DATA_PATH = "../input/tweet-sentiment-extraction/"
PKL_PATH = "../input/tweet-char-lvl-preds/"

K = 5
NUM_WORKERS = 4


# In[ ]:


df_test = pd.read_csv(DATA_PATH + 'test.csv').fillna('')
df_test['selected_text'] = ''
sub = pd.read_csv(DATA_PATH + 'sample_submission.csv')


# In[ ]:


MODELS = [
    ('bert-base-uncased-', 'theo'),
    ('bert-wwm-neutral-', 'theo'),
    ("roberta-", 'hk'),
    ("distil_", 'hk'),
    ("large_", 'hk'),
]

add_spaces_to = ["bert_", 'xlnet_', 'electra_', 'bertweet-']


# # Retrieveing 1st level model outputs

# ## Test predictions

# For inference on the private set, I use some of the first level scripts to retrieve the models. I only use a few models here, for faster inference time.

# ### DistilRoberta

# In[ ]:


get_ipython().run_cell_magic('time', '', '!python ../input/distil-roberta/infer.py')


# ### Roberta

# In[ ]:


get_ipython().run_cell_magic('time', '', '!python ../input/roberta-base/infer.py')


# ### Roberta-large

# In[ ]:


get_ipython().run_cell_magic('time', '', '!python ../input/roberta-large-code/infer.py')


# ### Bert-large-wwm

# In[ ]:


get_ipython().run_cell_magic('time', '', '!python ../input/tweet-inference-scripts/inference_bert_wwm.py')


# ### Bert-base

# In[ ]:


get_ipython().run_cell_magic('time', '', '!python ../input/tweet-inference-scripts/inference_bert_base.py')


# ### Retrieve everything

# In[ ]:


def create_input_data(models):
    char_pred_test_starts = []
    char_pred_test_ends = []

    for model, _ in models:
        with open(model + 'char_pred_test_start.pkl', "rb") as fp:   #Pickling
            probas = pickle.load(fp)  

            if model in add_spaces_to:
                probas = [np.concatenate([np.array([0]), p]) for p in probas]

            char_pred_test_starts.append(probas)

        with open(model + 'char_pred_test_end.pkl', "rb") as fp:   #Pickling
            probas = pickle.load(fp)

            if model in add_spaces_to:
                probas = [np.concatenate([np.array([0]), p]) for p in probas]

            char_pred_test_ends.append(probas)
            
    char_pred_test_start = [np.concatenate([char_pred_test_starts[m][i][:, np.newaxis] for m in range(len(models))], 
                                           1) for i in range(len(char_pred_test_starts[0]))]

    char_pred_test_end = [np.concatenate([char_pred_test_ends[m][i][:, np.newaxis] for m in range(len(models))], 
                                         1) for i in range(len(char_pred_test_starts[0]))]
    
    return char_pred_test_start, char_pred_test_end


# In[ ]:


char_pred_test_start, char_pred_test_end = create_input_data(MODELS)


# ## Oof predictions
# Because each of us used different folds, I have to make sure everything is put back in order.

# In[ ]:


def reorder(order_source, order_target, preds):
#     assert len(order_source) == len(order_target) and len(order_target) == len(preds)
    order_source = list(order_source)
    new_preds = []
    for tgt_idx in order_target:
        new_idx = order_source.index(tgt_idx)
        new_preds.append(preds[new_idx])
        
    return new_preds


df_train = pd.read_csv(DATA_PATH + 'train.csv').dropna().reset_index(drop=True)
df_train = df_train.sample(frac=1, random_state=SEED).reset_index(drop=True)
order_t = list(df_train['textID'].values)

df_train = pd.read_csv(DATA_PATH + 'train.csv').dropna()
df_train = df_train.sample(frac=1, random_state=50898).reset_index(drop=True)
order_hk = list(df_train['textID'].values)

ORDERS = {
    'theo': order_t,
    'hk': order_hk,
}


# In[ ]:


char_pred_oof_starts = []
char_pred_oof_ends = []

for model, author in tqdm(MODELS):
    with open(PKL_PATH + model + 'char_pred_oof_start.pkl', "rb") as fp:   #Pickling
        probas = pickle.load(fp)
        
        if author != 'hk':
            probas = reorder(ORDERS[author], ORDERS['hk'], probas)
        
        if model in add_spaces_to:
            probas = [np.concatenate([np.array([0]), p]) for p in probas]
            
        char_pred_oof_starts.append(probas)

    with open(PKL_PATH + model + 'char_pred_oof_end.pkl', "rb") as fp:   #Pickling
        probas = pickle.load(fp)
        
        if model in add_spaces_to:
            probas = [np.concatenate([np.array([0]), p]) for p in probas]
        
        if author != 'hk':
            probas = reorder(ORDERS[author], ORDERS['hk'], probas)
            
        char_pred_oof_ends.append(probas)


# In[ ]:


n_models = len(MODELS)

char_pred_oof_start = [np.concatenate([char_pred_oof_starts[m][i][:, np.newaxis] for m in range(n_models)], 
                                      1) for i in range(len(df_train))]

char_pred_oof_end = [np.concatenate([char_pred_oof_ends[m][i][:, np.newaxis] for m in range(n_models)], 
                                      1) for i in range(len(df_train))]


# In[ ]:


preds = {
    'test_start': np.array(char_pred_test_start),
    'test_end': np.array(char_pred_test_end),
    'oof_start': np.array(char_pred_oof_start),
    'oof_end': np.array(char_pred_oof_end),
}

model_names = [a + ' : ' + m for m, a in MODELS]
combs = [model_names]

print('Using models : ', combs)


# ## Text Data

# In[ ]:


tokenizer = Tokenizer(num_words=None, char_level=True, oov_token='UNK', lower=True)
tokenizer.fit_on_texts(df_train['text'].values)
len_voc = len(tokenizer.word_index) + 1

X_train = tokenizer.texts_to_sequences(df_train['text'].values)
X_test = tokenizer.texts_to_sequences(df_test['text'].values)


# ## Dataset
# - The Dataset is similar to the one shared in public kernels, but adapted to our character level network

# In[ ]:


def get_start_end_string(text, selected_text):
    len_selected_text = len(selected_text)
    idx_start, idx_end = 0, 0
    
    candidates_idx = [i for i, e in enumerate(text) if e == selected_text[0]]
    for idx in candidates_idx:
        if text[idx : idx + len_selected_text] == selected_text:
            idx_start = idx
            idx_end = idx + len_selected_text
            break
    assert text[idx_start: idx_end] == selected_text, f'"{text[idx_start: idx_end]}" instead of "{selected_text}" in "{text}"'

    char_targets = np.zeros(len(text))
    char_targets[idx_start: idx_end] = 1
    
    return idx_start, idx_end


# In[ ]:


class TweetCharDataset(Dataset):
    def __init__(self, df, X, start_probas, end_probas, n_models=1, max_len=150, train=True):
        self.max_len = max_len

        self.X = pad_sequences(X, maxlen=max_len, padding='post', truncating='post')
        
        self.start_probas = np.zeros((len(df), max_len, n_models), dtype=float)
        for i, p in enumerate(start_probas):
            len_ = min(len(p), max_len)
            self.start_probas[i, :len_] = p[:len_]

        self.end_probas = np.zeros((len(df), max_len, n_models), dtype=float)
        for i, p in enumerate(end_probas):
            len_ = min(len(p), max_len)
            self.end_probas[i, :len_] = p[:len_]
            
        self.sentiments_list = ['positive', 'neutral', 'negative']
        
        self.texts = df['text'].values
        self.selected_texts = df['selected_text'].values if train else [''] * len(df)
        self.sentiments = df['sentiment'].values
        self.sentiments_input = [self.sentiments_list.index(s) for s in self.sentiments]
        
        # Targets
        self.seg_label = np.zeros((len(df), max_len))
        
        if train:
            self.start_idx = []
            self.end_idx = []
            for i, (text, sel_text) in enumerate(zip(df['text'].values, df['selected_text'].values)):
                start, end = get_start_end_string(text, sel_text.strip())
                self.start_idx.append(start)
                self.end_idx.append(end)
                self.seg_label[i, start:end] = 1
        else:
            self.start_idx = [0] * len(df)
            self.end_idx = [0] * len(df)
        

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'ids': torch.tensor(self.X[idx], dtype=torch.long),
            'probas_start': torch.tensor(self.start_probas[idx]).float(),
            'probas_end': torch.tensor(self.end_probas[idx]).float(),
            'target_start': torch.tensor(self.start_idx[idx], dtype=torch.long),
            'target_end': torch.tensor(self.end_idx[idx], dtype=torch.long),
            'text': self.texts[idx],
            'selected_text': self.selected_texts[idx],
            'sentiment': self.sentiments[idx],
            'sentiment_input': torch.tensor(self.sentiments_input[idx]),
            'seg_label': torch.tensor(self.seg_label[idx])
        }


# ## Loss
# - We use the cross-entropy loss with label smoothing.

# In[ ]:


def ce_loss(
    pred, truth, smoothing=False, neighbour_smoothing=False, trg_pad_idx=-1, eps=0.1
):
    truth = truth.contiguous().view(-1)

    one_hot = torch.zeros_like(pred).scatter(1, truth.view(-1, 1), 1)
    one_hot_ = one_hot.clone()

    if smoothing:
        n_class = pred.size(1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)

        if neighbour_smoothing:
            n = 1
            for i in range(1, n):
                one_hot[:, :-i] += ((n - i) * eps) * one_hot_[:, i:]
                one_hot[:, i:] += ((n - i) * eps) * one_hot_[:, :-i]
            one_hot = one_hot / one_hot.sum(1, keepdim=True)

    loss = -one_hot * F.log_softmax(pred, dim=1)

    if trg_pad_idx >= 0:
        loss = loss.sum(dim=1)
        non_pad_mask = truth.ne(trg_pad_idx)
        loss = loss.masked_select(non_pad_mask)

    return loss.sum()


# In[ ]:


def loss_fn(start_logits, end_logits, start_positions, end_positions, config):

    bs = start_logits.size(0)

    start_loss = ce_loss(
        start_logits,
        start_positions,
        smoothing=config["smoothing"],
        eps=config["eps"],
        neighbour_smoothing=config["neighbour_smoothing"],
    )

    end_loss = ce_loss(
        end_logits,
        end_positions,
        smoothing=config["smoothing"],
        eps=config["eps"],
        neighbour_smoothing=config["neighbour_smoothing"],
    )

    total_loss = start_loss + end_loss

    return total_loss / bs


# ## Metric

# In[ ]:


def jaccard_from_logits_string(data, start_logits, end_logits):
    
    n = start_logits.size(0)
    score = 0

    start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
    end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

    for i in range(n):
        start_idx = np.argmax(start_logits[i])
        end_idx = np.argmax(end_logits[i])
        text = data["text"][i]
        pred = text[start_idx: end_idx]

        score += jaccard(data["selected_text"][i], pred)

    return score


# In[ ]:


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    try:
        return float(len(c)) / (len(a) + len(b) - len(c))
    except:
        return 0


# ## Predict

# In[ ]:


def predict(model, dataset, batch_size=32):
    model.eval()
    start_probas = []
    end_probas = []

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )

    with torch.no_grad():
        for data in loader:
            start_logits, end_logits = model(
                data["ids"].cuda(), 
                data['sentiment_input'].cuda(), 
                data['probas_start'].cuda(), 
                data['probas_end'].cuda()
            )

            start_probs = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

            for s, e in zip(start_probs, end_probs):
                start_probas.append(list(s))
                end_probas.append(list(e))

    return start_probas, end_probas


# ## SWA
# > From [torchcontrib](https://github.com/pytorch/contrib/blob/0b8e4271812e8849232f2e2bb6ee129393162d57/torchcontrib/optim/swa.py)

# In[ ]:


from collections import defaultdict
from itertools import chain
from torch.optim import Optimizer
import torch
import warnings


class SWA(Optimizer):
    def __init__(self, optimizer, swa_start=None, swa_freq=None, swa_lr=None):
        r"""Implements Stochastic Weight Averaging (SWA).
        Stochastic Weight Averaging was proposed in `Averaging Weights Leads to
        Wider Optima and Better Generalization`_ by Pavel Izmailov, Dmitrii
        Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson
        (UAI 2018).
        SWA is implemented as a wrapper class taking optimizer instance as input
        and applying SWA on top of that optimizer.
        SWA can be used in two modes: automatic and manual. In the automatic
        mode SWA running averages are automatically updated every
        :attr:`swa_freq` steps after :attr:`swa_start` steps of optimization. If
        :attr:`swa_lr` is provided, the learning rate of the optimizer is reset
        to :attr:`swa_lr` at every step starting from :attr:`swa_start`. To use
        SWA in automatic mode provide values for both :attr:`swa_start` and
        :attr:`swa_freq` arguments.
        Alternatively, in the manual mode, use :meth:`update_swa` or
        :meth:`update_swa_group` methods to update the SWA running averages.
        In the end of training use `swap_swa_sgd` method to set the optimized
        variables to the computed averages.
        Args:
            optimizer (torch.optim.Optimizer): optimizer to use with SWA
            swa_start (int): number of steps before starting to apply SWA in
                automatic mode; if None, manual mode is selected (default: None)
            swa_freq (int): number of steps between subsequent updates of
                SWA running averages in automatic mode; if None, manual mode is
                selected (default: None)
            swa_lr (float): learning rate to use starting from step swa_start
                in automatic mode; if None, learning rate is not changed
                (default: None)
        Examples:
            >>> # automatic mode
            >>> base_opt = torch.optim.SGD(model.parameters(), lr=0.1)
            >>> opt = torchcontrib.optim.SWA(
            >>>                 base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)
            >>> for _ in range(100):
            >>>     opt.zero_grad()
            >>>     loss_fn(model(input), target).backward()
            >>>     opt.step()
            >>> opt.swap_swa_sgd()
            >>> # manual mode
            >>> opt = torchcontrib.optim.SWA(base_opt)
            >>> for i in range(100):
            >>>     opt.zero_grad()
            >>>     loss_fn(model(input), target).backward()
            >>>     opt.step()
            >>>     if i > 10 and i % 5 == 0:
            >>>         opt.update_swa()
            >>> opt.swap_swa_sgd()
        .. note::
            SWA does not support parameter-specific values of :attr:`swa_start`,
            :attr:`swa_freq` or :attr:`swa_lr`. In automatic mode SWA uses the
            same :attr:`swa_start`, :attr:`swa_freq` and :attr:`swa_lr` for all
            parameter groups. If needed, use manual mode with
            :meth:`update_swa_group` to use different update schedules for
            different parameter groups.
        .. note::
            Call :meth:`swap_swa_sgd` in the end of training to use the computed
            running averages.
        .. note::
            If you are using SWA to optimize the parameters of a Neural Network
            containing Batch Normalization layers, you need to update the
            :attr:`running_mean` and :attr:`running_var` statistics of the
            Batch Normalization module. You can do so by using
            `torchcontrib.optim.swa.bn_update` utility.
        .. note::
            See the blogpost
            https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/
            for an extended description of this SWA implementation.
        .. note::
            The repo https://github.com/izmailovpavel/contrib_swa_examples
            contains examples of using this SWA implementation.
        .. _Averaging Weights Leads to Wider Optima and Better Generalization:
            https://arxiv.org/abs/1803.05407
        .. _Improving Consistency-Based Semi-Supervised Learning with Weight
            Averaging:
            https://arxiv.org/abs/1806.05594
        """
        self._auto_mode, (self.swa_start, self.swa_freq) =             self._check_params(self, swa_start, swa_freq)
        self.swa_lr = swa_lr

        if self._auto_mode:
            if swa_start < 0:
                raise ValueError("Invalid swa_start: {}".format(swa_start))
            if swa_freq < 1:
                raise ValueError("Invalid swa_freq: {}".format(swa_freq))
        else:
            if self.swa_lr is not None:
                warnings.warn(
                    "Some of swa_start, swa_freq is None, ignoring swa_lr")
            # If not in auto mode make all swa parameters None
            self.swa_lr = None
            self.swa_start = None
            self.swa_freq = None

        if self.swa_lr is not None and self.swa_lr < 0:
            raise ValueError("Invalid SWA learning rate: {}".format(swa_lr))

        self.optimizer = optimizer

        self.defaults = self.optimizer.defaults
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.opt_state = self.optimizer.state
        for group in self.param_groups:
            group['n_avg'] = 0
            group['step_counter'] = 0

    @staticmethod
    def _check_params(self, swa_start, swa_freq):
        params = [swa_start, swa_freq]
        params_none = [param is None for param in params]
        if not all(params_none) and any(params_none):
            warnings.warn(
                "Some of swa_start, swa_freq is None, ignoring other")
        for i, param in enumerate(params):
            if param is not None and not isinstance(param, int):
                params[i] = int(param)
                warnings.warn("Casting swa_start, swa_freq to int")
        return not any(params_none), params

    def _reset_lr_to_swa(self):
        if self.swa_lr is None:
            return
        for param_group in self.param_groups:
            if param_group['step_counter'] >= self.swa_start:
                param_group['lr'] = self.swa_lr

    def update_swa_group(self, group):
        r"""Updates the SWA running averages for the given parameter group.
        Arguments:
            param_group (dict): Specifies for what parameter group SWA running
                averages should be updated
        Examples:
            >>> # automatic mode
            >>> base_opt = torch.optim.SGD([{'params': [x]},
            >>>             {'params': [y], 'lr': 1e-3}], lr=1e-2, momentum=0.9)
            >>> opt = torchcontrib.optim.SWA(base_opt)
            >>> for i in range(100):
            >>>     opt.zero_grad()
            >>>     loss_fn(model(input), target).backward()
            >>>     opt.step()
            >>>     if i > 10 and i % 5 == 0:
            >>>         # Update SWA for the second parameter group
            >>>         opt.update_swa_group(opt.param_groups[1])
            >>> opt.swap_swa_sgd()
        """
        for p in group['params']:
            param_state = self.state[p]
            if 'swa_buffer' not in param_state:
                param_state['swa_buffer'] = torch.zeros_like(p.data)
            buf = param_state['swa_buffer']
            virtual_decay = 1 / float(group["n_avg"] + 1)
            diff = (p.data - buf) * virtual_decay
            buf.add_(diff)
        group["n_avg"] += 1

    def update_swa(self):
        r"""Updates the SWA running averages of all optimized parameters.
        """
        for group in self.param_groups:
            self.update_swa_group(group)

    def swap_swa_sgd(self):
        r"""Swaps the values of the optimized variables and swa buffers.
        It's meant to be called in the end of training to use the collected
        swa running averages. It can also be used to evaluate the running
        averages during training; to continue training `swap_swa_sgd`
        should be called again.
        """
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'swa_buffer' not in param_state:
                    # If swa wasn't applied we don't swap params
                    warnings.warn(
                        "SWA wasn't applied to param {}; skipping it".format(p))
                    continue
                buf = param_state['swa_buffer']
                tmp = torch.empty_like(p.data)
                tmp.copy_(p.data)
                p.data.copy_(buf)
                buf.copy_(tmp)

    def step(self, closure=None):
        r"""Performs a single optimization step.
        In automatic mode also updates SWA running averages.
        """
        self._reset_lr_to_swa()
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            group["step_counter"] += 1
            steps = group["step_counter"]
            if self._auto_mode:
                if steps > self.swa_start and steps % self.swa_freq == 0:
                    self.update_swa_group(group)
        return loss

    def state_dict(self):
        r"""Returns the state of SWA as a :class:`dict`.
        It contains three entries:
            * opt_state - a dict holding current optimization state of the base
                optimizer. Its content differs between optimizer classes.
            * swa_state - a dict containing current state of SWA. For each
                optimized variable it contains swa_buffer keeping the running
                average of the variable
            * param_groups - a dict containing all parameter groups
        """
        opt_state_dict = self.optimizer.state_dict()
        swa_state = {(id(k) if isinstance(k, torch.Tensor) else k): v
                     for k, v in self.state.items()}
        opt_state = opt_state_dict["state"]
        param_groups = opt_state_dict["param_groups"]
        return {"opt_state": opt_state, "swa_state": swa_state,
                "param_groups": param_groups}

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.
        Args:
            state_dict (dict): SWA optimizer state. Should be an object returned
                from a call to `state_dict`.
        """
        swa_state_dict = {"state": state_dict["swa_state"],
                          "param_groups": state_dict["param_groups"]}
        opt_state_dict = {"state": state_dict["opt_state"],
                          "param_groups": state_dict["param_groups"]}
        super(SWA, self).load_state_dict(swa_state_dict)
        self.optimizer.load_state_dict(opt_state_dict)
        self.opt_state = self.optimizer.state

    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.
        This can be useful when fine tuning a pre-trained network as frozen
        layers can be made trainable and added to the :class:`Optimizer` as
        training progresses.
        Args:
            param_group (dict): Specifies what Tensors should be optimized along
            with group specific optimization options.
        """
        param_group['n_avg'] = 0
        param_group['step_counter'] = 0
        self.optimizer.add_param_group(param_group)

    @staticmethod
    def bn_update(loader, model, device=None):
        r"""Updates BatchNorm running_mean, running_var buffers in the model.
        It performs one pass over data in `loader` to estimate the activation
        statistics for BatchNorm layers in the model.
        Args:
            loader (torch.utils.data.DataLoader): dataset loader to compute the
                activation statistics on. Each data batch should be either a
                tensor, or a list/tuple whose first element is a tensor
                containing data.
            model (torch.nn.Module): model for which we seek to update BatchNorm
                statistics.
            device (torch.device, optional): If set, data will be trasferred to
                :attr:`device` before being passed into :attr:`model`.
        """
        if not _check_bn(model):
            return
        was_training = model.training
        model.train()
        momenta = {}
        model.apply(_reset_bn)
        model.apply(lambda module: _get_momenta(module, momenta))
        n = 0
        for input in loader:
            if isinstance(input, (list, tuple)):
                input = input[0]
            b = input.size(0)

            momentum = b / float(n + b)
            for module in momenta.keys():
                module.momentum = momentum

            if device is not None:
                input = input.to(device)

            model(input)
            n += b

        model.apply(lambda module: _set_momenta(module, momenta))
        model.train(was_training)


# BatchNorm utils
def _check_bn_apply(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def _check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn_apply(module, flag))
    return flag[0]


def _reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


# ## Fit

# In[ ]:


def fit(
    model,
    train_dataset,
    val_dataset,
    loss_config,
    epochs=5,
    swa_first_epoch=5,
    batch_size=8,
    acc_steps=1,
    weight_decay=0,
    warmup_prop=0.0,
    lr=5e-4,
    cp=False,
    use_len_sampler=True,
):
    best_jac = 0
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )

    optimizer = Adam(model.parameters(), lr=lr) #, betas=(0.5, 0.999))
    optimizer = SWA(optimizer)

    n_steps = float(epochs * len(train_loader)) / float(acc_steps)
    num_warmup_steps = int(warmup_prop * n_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, n_steps
    )

    total_steps = 0
    for epoch in range(epochs):
        model.train()
        start_time = time.time()

        optimizer.zero_grad()
        avg_loss = 0

        for step, data in enumerate(train_loader):
            total_steps += 1
            start_logits, end_logits = model(
                data["ids"].cuda(), 
                data['sentiment_input'].cuda(), 
                data['probas_start'].cuda(), 
                data['probas_end'].cuda()
            )

            loss = loss_fn(
                start_logits,
                end_logits,
                data["target_start"].cuda(),
                data["target_end"].cuda(),
                config=loss_config,
            )

            avg_loss += loss.item() / len(train_loader)
            loss.backward()

            if (step + 1) % acc_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
        model.eval()
        avg_val_loss = 0.0
        val_jac = 0.0

        if epoch + 1 >= swa_first_epoch:
            optimizer.update_swa()
            optimizer.swap_swa_sgd()

        with torch.no_grad():
            for data in val_loader:
                
                start_logits, end_logits = model(
                    data["ids"].cuda(), 
                    data['sentiment_input'].cuda(), 
                    data['probas_start'].cuda(), 
                    data['probas_end'].cuda()
                )

                loss = loss_fn(
                    start_logits.detach(),
                    end_logits.detach(),
                    data["target_start"].cuda().detach(),
                    data["target_end"].cuda().detach(),
                    config=loss_config,
                )

                avg_val_loss += loss.item() / len(val_loader)

                val_jac += jaccard_from_logits_string(data, start_logits, end_logits) / len(
                    val_dataset
                )
        
        if epoch + 1 >= swa_first_epoch:
            optimizer.swap_swa_sgd()
            
        if val_jac >= best_jac and cp:
            save_model_weights(model, "checkpoint.pt", verbose=0)
            best_jac = val_jac

        dt = time.time() - start_time
        lr = scheduler.get_lr()[0]
        print(f"Epoch {epoch + 1}/{epochs} \t lr={lr:.1e} \t t={dt:.0f}s \t", end="")
        print(
            f"loss={avg_loss:.3f} \t val_loss={avg_val_loss:.3f} \t val_jaccard={val_jac:.4f}"
        )

    del loss, data, avg_val_loss, avg_loss, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    
    if epoch >= swa_first_epoch: # back to swa weights
        optimizer.swap_swa_sgd()

    return best_jac if cp else val_jac


# # Models
# We have three models : 
# - A RNN
# - A 1D-CNN
# - A Wavenet

# ## Modules

# In[ ]:


import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding="same", use_bn=True):
        super().__init__()
        if padding == "same":
            padding = kernel_size // 2 * dilation
        
        if use_bn:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation),
                nn.ReLU(),
            )
                
    def forward(self, x):
        return self.conv(x)

class Waveblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=[1], padding="same"):
        super().__init__()
        self.n = len(dilations)
        
        if padding == "same":
            padding = kernel_size // 2
            
        self.init_conv = nn.Conv1d(in_channels, out_channels, 1)
        
        self.convs_tanh = nn.ModuleList([])
        self.convs_sigm = nn.ModuleList([])
        self.convs = nn.ModuleList([])
        
        for dilation in dilations:
            self.convs_tanh.append(
                nn.Sequential(
                    nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding*dilation, dilation=dilation),
                    nn.Tanh(),
                )
            )
            self.convs_sigm.append(
                nn.Sequential(
                    nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding*dilation, dilation=dilation),
                    nn.Sigmoid(),
                )
            )
            self.convs.append(nn.Conv1d(out_channels, out_channels, 1))
        
    def forward(self, x):
        x = self.init_conv(x)
        res_x = x
        
        for i in range(self.n):
            x_tanh = self.convs_tanh[i](x)
            x_sigm = self.convs_sigm[i](x)
            x = x_tanh * x_sigm
            x = self.convs[i](x)
            res_x = res_x + x
        
        return res_x


# ## RNN

# In[ ]:


class TweetCharModel(nn.Module):
    def __init__(self, len_voc, use_msd=True,
                 embed_dim=64, lstm_dim=64, char_embed_dim=32, sent_embed_dim=32, ft_lstm_dim=32, n_models=1):
        super().__init__()
        self.use_msd = use_msd
        
        self.char_embeddings = nn.Embedding(len_voc, char_embed_dim)
        self.sentiment_embeddings = nn.Embedding(3, sent_embed_dim)
        
        self.proba_lstm = nn.LSTM(n_models * 2, ft_lstm_dim, batch_first=True, bidirectional=True)
        
        self.lstm = nn.LSTM(char_embed_dim + ft_lstm_dim * 2 + sent_embed_dim, lstm_dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(lstm_dim * 2, lstm_dim, batch_first=True, bidirectional=True)

        self.logits = nn.Sequential(
            nn.Linear(lstm_dim *  4, lstm_dim),
            nn.ReLU(),
            nn.Linear(lstm_dim, 2),
        )
        
        self.high_dropout = nn.Dropout(p=0.5)
    
    def forward(self, tokens, sentiment, start_probas, end_probas):
        bs, T = tokens.size()
        
        probas = torch.cat([start_probas, end_probas], -1)
        probas_fts, _ = self.proba_lstm(probas)

        char_fts = self.char_embeddings(tokens)
        
        sentiment_fts = self.sentiment_embeddings(sentiment).view(bs, 1, -1)
        sentiment_fts = sentiment_fts.repeat((1, T, 1))
        
        features = torch.cat([char_fts, sentiment_fts, probas_fts], -1)
        features, _ = self.lstm(features)
        features2, _ = self.lstm2(features)
        
        features = torch.cat([features, features2], -1)
        
        if self.use_msd and self.training:
            logits = torch.mean(
                torch.stack(
                    [self.logits(self.high_dropout(features)) for _ in range(5)],
                    dim=0,
                    ),
                dim=0,
            )
        else:
            logits = self.logits(features)

        start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]

        return start_logits, end_logits


# ## Wavenet

# In[ ]:


class WaveNet(nn.Module):
    def __init__(self, len_voc, use_msd=True, dilations=[1], 
                 cnn_dim=64, char_embed_dim=32, sent_embed_dim=32, proba_cnn_dim=32, n_models=1, kernel_size=3, use_bn=True):
        super().__init__()
        self.use_msd = use_msd
        
        self.char_embeddings = nn.Embedding(len_voc, char_embed_dim)
        self.sentiment_embeddings = nn.Embedding(3, sent_embed_dim)
        
        self.probas_cnn = ConvBlock(n_models * 2, proba_cnn_dim, kernel_size=kernel_size, use_bn=use_bn)
         
        self.cnn = nn.Sequential(
            Waveblock(char_embed_dim + sent_embed_dim + proba_cnn_dim, cnn_dim, kernel_size=kernel_size, dilations=dilations),
            nn.BatchNorm1d(cnn_dim),
            Waveblock(cnn_dim, cnn_dim * 2, kernel_size=kernel_size, dilations=dilations),
            nn.BatchNorm1d(cnn_dim * 2),
            Waveblock(cnn_dim * 2 , cnn_dim * 4, kernel_size=kernel_size, dilations=dilations),
            nn.BatchNorm1d(cnn_dim * 4),
        )
        
        self.logits = nn.Sequential(
            nn.Linear(cnn_dim * 4, cnn_dim),
            nn.ReLU(),
            nn.Linear(cnn_dim, 2),
        )
        
        self.high_dropout = nn.Dropout(p=0.5)
        
    def forward(self, tokens, sentiment, start_probas, end_probas):
        bs, T = tokens.size()
        
        probas = torch.cat([start_probas, end_probas], -1).permute(0, 2, 1)
        probas_fts = self.probas_cnn(probas).permute(0, 2, 1)

        char_fts = self.char_embeddings(tokens)
        
        sentiment_fts = self.sentiment_embeddings(sentiment).view(bs, 1, -1)
        sentiment_fts = sentiment_fts.repeat((1, T, 1))
        
        x = torch.cat([char_fts, sentiment_fts, probas_fts], -1).permute(0, 2, 1)

        features = self.cnn(x).permute(0, 2, 1) # [Bs x T x nb_ft]
    
        if self.use_msd and self.training:
            logits = torch.mean(
                torch.stack(
                    [self.logits(self.high_dropout(features)) for _ in range(5)],
                    dim=0,
                    ),
                dim=0,
            )
        else:
            logits = self.logits(features)

        start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]

        return start_logits, end_logits


# ## CNN

# In[ ]:


class ConvNet(nn.Module):
    def __init__(self, len_voc, use_msd=True,
                 cnn_dim=64, char_embed_dim=32, sent_embed_dim=32, proba_cnn_dim=32, n_models=1, kernel_size=3, use_bn=False):
        super().__init__()
        self.use_msd = use_msd
        
        self.char_embeddings = nn.Embedding(len_voc, char_embed_dim)
        self.sentiment_embeddings = nn.Embedding(3, sent_embed_dim)
        
        self.probas_cnn = ConvBlock(n_models * 2, proba_cnn_dim, kernel_size=kernel_size, use_bn=use_bn)
         
        self.cnn = nn.Sequential(
            ConvBlock(char_embed_dim + sent_embed_dim + proba_cnn_dim, cnn_dim, kernel_size=kernel_size, use_bn=use_bn),
            ConvBlock(cnn_dim, cnn_dim * 2, kernel_size=kernel_size, use_bn=use_bn),
            ConvBlock(cnn_dim * 2 , cnn_dim * 4, kernel_size=kernel_size, use_bn=use_bn),
            ConvBlock(cnn_dim * 4, cnn_dim * 8, kernel_size=kernel_size, use_bn=use_bn),
        )
        
        self.logits = nn.Sequential(
            nn.Linear(cnn_dim * 8, cnn_dim),
            nn.ReLU(),
            nn.Linear(cnn_dim, 2),
        )
        
        self.high_dropout = nn.Dropout(p=0.5)
        
    def forward(self, tokens, sentiment, start_probas, end_probas):
        bs, T = tokens.size()
        
        probas = torch.cat([start_probas, end_probas], -1).permute(0, 2, 1)
        probas_fts = self.probas_cnn(probas).permute(0, 2, 1)

        char_fts = self.char_embeddings(tokens)
        
        sentiment_fts = self.sentiment_embeddings(sentiment).view(bs, 1, -1)
        sentiment_fts = sentiment_fts.repeat((1, T, 1))
        
        x = torch.cat([char_fts, sentiment_fts, probas_fts], -1).permute(0, 2, 1)

        features = self.cnn(x).permute(0, 2, 1) # [Bs x T x nb_ft]
    
        if self.use_msd and self.training:
            logits = torch.mean(
                torch.stack(
                    [self.logits(self.high_dropout(features)) for _ in range(5)],
                    dim=0,
                    ),
                dim=0,
            )
        else:
            logits = self.logits(features)

        start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]

        return start_logits, end_logits


# # $k$-fold

# In[ ]:


def k_fold(config, df_train, df_test, X_train, X_test, preds, len_voc, k=5, seed=42, save=True, model_name='model'):
    time = str(datetime.datetime.now())[:16]
    score = 0
    splits = list(StratifiedKFold(n_splits=k, random_state=seed).split(X=df_train, y=df_train['sentiment']))
    
    pred_oof = [[[], []] for i in range(len(df_train))]
    pred_tests = [] 
    
    test_dataset = TweetCharDataset(
        df_test, X_test, preds['test_start'], preds['test_end'], 
        max_len=config.max_len_val, train=False, n_models=config.n_models
    )
    
    for i, (train_idx, val_idx) in enumerate(splits):
        print(f"\n-------------   Fold {i + 1}  -------------")
        seed_everything(seed + i)

        if config.model == 'rnn':
            model = TweetCharModel(
                len_voc,
                use_msd=config.use_msd, 
                n_models=config.n_models,   
                lstm_dim=config.lstm_dim,
                ft_lstm_dim=config.ft_lstm_dim,
                char_embed_dim=config.char_embed_dim,
                sent_embed_dim=config.sent_embed_dim,
            ).cuda()
        elif config.model == 'cnn':
            model = ConvNet(
                len_voc,
                use_msd=config.use_msd, 
                n_models=config.n_models,  
                use_bn=config.use_bn,
                cnn_dim=config.cnn_dim,
                proba_cnn_dim=config.proba_cnn_dim,
                char_embed_dim=config.char_embed_dim,
                sent_embed_dim=config.sent_embed_dim,
                kernel_size=config.kernel_size,
            ).cuda()
        else:
            model = WaveNet(
                len_voc,
                use_msd=config.use_msd, 
                n_models=config.n_models,  
                use_bn=config.use_bn,
                cnn_dim=config.cnn_dim,
                proba_cnn_dim=config.proba_cnn_dim,
                char_embed_dim=config.char_embed_dim,
                sent_embed_dim=config.sent_embed_dim,
                kernel_size=config.kernel_size,
                dilations=config.dilations, 
            ).cuda()
        
        model.zero_grad()

        train_dataset = TweetCharDataset(
            df_train.iloc[train_idx],
            X_train[train_idx],
            preds['oof_start'][train_idx],
            preds['oof_end'][train_idx],
            max_len=config.max_len,
            n_models=config.n_models,
        )
        
        val_dataset = TweetCharDataset(
            df_train.iloc[val_idx], 
            X_train[val_idx], 
            preds['oof_start'][val_idx], 
            preds['oof_end'][val_idx],
            max_len=config.max_len_val,
            n_models=config.n_models,
        )
        
        print('\n- Training \n')

        fold_score = fit(
            model, 
            train_dataset, 
            val_dataset, 
            config.loss_config,
            epochs=config.epochs, 
            batch_size=config.batch_size, 
            lr=config.lr, 
            warmup_prop=config.warmup_prop,
            swa_first_epoch=config.swa_first_epoch,
            use_len_sampler=config.use_len_sampler,
            cp=False
        )
        
        score += fold_score / k

        print('\n- Predicting ')

        pred_val_start, pred_val_end = predict(model, val_dataset, batch_size=config.batch_size_val)
        for j, idx in enumerate(val_idx):
            pred_oof[idx] = [pred_val_start[j], pred_val_end[j]]
        
        pred_test = predict(model, test_dataset, batch_size=config.batch_size_val)
        pred_tests.append(pred_test)            

        del model, train_dataset, val_dataset
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f'\n Local CV jaccard is {score:.4f}')
    return pred_oof, pred_tests


# # Configs

# ## Wavenet

# In[ ]:


class ConfigWav:
    model = 'wavenet'
    n_models = len(MODELS)
    
    # Texts
    max_len = 150
    max_len_val = 150
    
    # Architecture
    sent_embed_dim = 16
    char_embed_dim = 16
    proba_cnn_dim = 16
    kernel_size = 3
    dilations = [1]
    
    cnn_dim = 32
    
    use_bn = True
    use_msd = True
    
    # Loss function
    loss_config = {
        "smoothing": True,
        "neighbour_smoothing": False,
        "eps": 0.1,
        "use_dist_loss": False,
        "dist_loss_weight": 1,
    }
    
    # Training
    use_len_sampler = False
    
    batch_size = 128
    batch_size_val = 512
    
    epochs = 5
    swa_first_epoch = 5
    lr = 4e-3
    warmup_prop = 0.
    
    # Post-processing
    remove_neutral = False
    
    # PL
    pl_confidence = 0.35


# ## CNN

# In[ ]:


class ConfigCNN:
    model = 'cnn'
    n_models = len(MODELS)
    
    # Texts
    max_len = 150
    max_len_val = 150
    
    # Architecture
    sent_embed_dim = 16
    char_embed_dim = 16
    proba_cnn_dim = 16
    kernel_size = 3
    
    cnn_dim = 32
    
    use_bn = True
    use_msd = True
    
    # Loss function
    loss_config = {
        "smoothing": True,
        "neighbour_smoothing": False,
        "eps": 0.1,
        "use_dist_loss": False,
        "dist_loss_weight": 1,
    }
    
    # Training
    use_len_sampler = False
    
    batch_size = 128
    batch_size_val = 512

    epochs = 5
    swa_first_epoch = 5
    lr = 4e-3
    warmup_prop = 0.

    # Post-processing
    remove_neutral = False
    
    pl_confidence = 0.35


# ## RNN

# In[ ]:


class ConfigRNN:
    model = 'rnn'
    n_models = len(MODELS)
    
    # Texts
    max_len = 150
    max_len_val = 150
    
    # Architecture
    sent_embed_dim = 16 # 32 works as well
    char_embed_dim = 8
    ft_lstm_dim = 16
    
    lstm_dim = 64
    use_msd = True
    
    # Loss function
    loss_config = {
        "smoothing": True,
        "neighbour_smoothing": False,
        "eps": 0.1,
        "use_dist_loss": False,
        "dist_loss_weight": 1,
    }
    
    # Training
    use_len_sampler = False
    
    batch_size = 128
    batch_size_val = 512

    epochs = 10
    swa_first_epoch = 5
    lr = 5e-3
    warmup_prop = 0.

    # Post-processing
    remove_neutral = False
    
    pl_confidence = 0.35


# # Train

# In[ ]:


configs = [ConfigWav(), ConfigRNN(), ConfigCNN()]


# In[ ]:


pred_oofs = []
pred_tests = []


# In[ ]:


for idx, comb in enumerate(combs):        
    print('#' * 80)
    print(f' -> Combination {idx + 1}/{len(combs)} : \n {" / ".join(list(comb))} ')
    print('#' * 80, "\n")
    used = [model_names.index(c) for c in comb]
    
    used_preds = {}
    for key in preds.keys():
        used_preds[key] = np.array([preds[key][i][:, used] for i in range(len(preds[key]))])
    
    for config in configs:
        
        print(f' -> Training {config.model.upper()}')
        
        config.n_models = len(used)
        pred_oof, pred_test = k_fold(config, df_train, df_test, np.array(X_train), np.array(X_test), used_preds, len_voc, 
                                      k=K, seed=SEED, model_name='wavenet_0_0')
        
        pred_oofs.append(pred_oof)
        pred_tests.append(pred_test)
        
        print('\n')


# ## Retrieving predictions

# In[ ]:


def string_from_preds_char_level(dataset, preds, test=False, remove_neutral=False, uncensored=False, cleaned=False):
    selected_texts = []
    n_models = len(preds)

    for idx in range(len(dataset)):
        data = dataset[idx]

        if test:
            start_probas = np.mean([preds[i][0][idx] for i in range(n_models)], 0)
            end_probas = np.mean([preds[i][1][idx] for i in range(n_models)], 0)
        else:
            start_probas = preds[idx][0]
            end_probas = preds[idx][1]

        start_idx = np.argmax(start_probas)
        end_idx = np.argmax(end_probas)

        if end_idx < start_idx:
            selected_text = data["text"]
        elif remove_neutral and data["sentiment"] == "neutral":
            selected_text = data["text"]
        else:
            selected_text = data["text"][start_idx: end_idx]

        selected_texts.append(selected_text.strip())

    return selected_texts


# In[ ]:


config = ConfigWav()
config.n_models = len(preds['oof_start'][0][0])

test_dataset = TweetCharDataset(
    df_test, X_test, preds['test_start'], preds['test_end'], 
    max_len=config.max_len_val, train=False, n_models=config.n_models, 
)

dataset = TweetCharDataset(
    df_train, X_train, preds['test_start'], preds['test_end'], 
    max_len=config.max_len_val, train=False, n_models=config.n_models, 
)


# In[ ]:


pred_oof = (np.array(pred_oofs[0]) + np.array(pred_oofs[1]) + np.array(pred_oofs[2])) / 3
pred_test = pred_tests[0] + pred_tests[1] + pred_tests[2]


# In[ ]:


selected_texts_oof = string_from_preds_char_level(dataset, pred_oof, test=False, remove_neutral=False)


# In[ ]:


scores = [jaccard(pred, truth) for (pred, truth) in zip(selected_texts_oof, df_train['selected_text'])]
score = np.mean(scores)
print(f'Local CV score is {score:.4f}')


# ## Submission

# In[ ]:


selected_texts = string_from_preds_char_level(test_dataset, pred_test, test=True, remove_neutral=False)


# In[ ]:


sub['selected_text'] = selected_texts  
sub.to_csv('submission.csv', index=False)
sub.head()

