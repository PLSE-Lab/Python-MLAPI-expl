#!/usr/bin/env python
# coding: utf-8

# # Pretraining knowledge priors with Transformers and adaptation to unseen tasks
# 
# This is an attempt to learn core knowledge priors for ARC using relational neural networks. A lot of fun working on this!
# - Build Transformer neural networks to encode and reason about the relations between objects among the input-output pairs.
# - Pretrain the model on the training set tasks to build representations for core knowledge priors. 
# - At test time, first train on the demostration examples of the unseen tasks, then predict the test output. 
# - So far, a pretrained model + 2-hr test-time training solves 9% of validation tasks and (only..) 1% of hidden test tasks (0.99 on LB). 
# - Longer pretraining time makes adaptation to unseen tasks more efficient. 

# ## Learning representations for ARC
# 
# For a neural network to reason about (unseen) ARC tasks, it likely needs a network structure with built-in inductive bias of core knowledge priors and the relational reasoning process, as well as large amounts of pretraining to encode the abstract knowledge, similar to a human's exposure to vast amount of early-life experiences before being able to solve these tasks.
# 
# ### - Graph networks and Transformers
# 
# As the ARC tasks involve a lot of reasoning of the relationships between objects, [graph networks](https://arxiv.org/pdf/1806.01261.pdf) seem to be a suitable for this problem. A recent work is [Abstract Diagrammatic Reasoning with Multiplex Graph Networks](https://openreview.net/forum?id=ByxQB1BKwH). 
# 
# However, detecting and encoding objects are not easy, as well as synthesizing the output image. For the relatively small grid sizes in ARC, we can think of every grid point as a putative object and use [Transformers](https://arxiv.org/abs/1706.03762) (and [non-local neural networks](https://arxiv.org/abs/1711.07971) in general) as graph-structured neural networks. An example of using transformers on image data is [Image Transformers](https://arxiv.org/abs/1802.05751). 
# 
# ### - Encoding input-output pairs and predicting the test output
# 
# Predicting the test output given the test input, conditioned on a set of context input-output pairs, is conceptually a regression problem. Methods such as
# [Neural Processes](https://arxiv.org/abs/1807.01622), [Attentive Neural Processes](https://arxiv.org/abs/1901.05761) produces a series of distributions conditioned on the context pairs to predict the test output. One difference is that each ARC task has only one correct output, so it's not clear if it will benefit from the probablistic nature of neural processes. But we can still use the idea of conditioning.
# 
# ### - Learning to solve the demostration examples before predicting the test output
# 
# Since we're given a few demonstration examples, we could train the neural network to predict the demonstration outputs of the unseen task before attempting on the test output. This is limited by the 2-hour GPU time of the Kaggle kernel. Thus, a faster learning curve here will give a better performance. (I trained all 100 tasks together. Maybe it's also a good idea to train on each task separately..)
# 
# ### - Pretraining to build representations for core knowledge priors
# 
# As ARC tasks are very hard, we can pretrain the model on the training set with data augmentations. This might help the model learn to better encode core knowledge priors and reason about common object relationships. The training set includes all training data and half of the evaluation data, the other half is used for validation. Of course, the pretrained model can also be directly used on the unseen test tasks without test-time training, but the accuracy is pretty low (validation 1~2%, LB 0%). When using a pretrained model (>8000 epochs) to initialize the weights for test-time training on demonstraction examples, we can increase the validation accuracy to 9% and get 1% on LB. I think longer pretraining with larger models on wide variaties of common sense visual tasks will help build better representations for core knowledge prior. I'm also interested in ideas from meta-learning, curriculum learning, etc, but haven't got a chance to try them yet. 
# 
# 
# 

# ## Model
# 
# ### - Encoding and summarization of demonstration input/output pairs
# 
# Each demonstraction (demo) input ($x_i^{d}$) or output ($y_i^{d}$) image first goes through an embedding layer, a convolution block (optional), and a 2D positional embedding layer before being flattened into 1D to produce $x_i^{d,enc}$, $y_i^{d,enc}$. 
# 
# Each encoded image then passes through a self-attention layer and a cross-attention layer (input attends to output, and output attends to input) to encode the relationships between objects within the input-output pair, producing $x_i^{d,attn}$, $y_i^{d,attn}$. 
# 
# Theses representation then goes through two paths. 
# 
# In one path, the input/output presentation are mean-aggregated across the node axis and across demo examples as a summarized representation of the task context, $x_i^{d,aggr} = mean(x_i^{d,attn})$, $y_i^{d,aggr} = mean(y_i^{d,attn})$, $x^{d,aggr} = \frac{1}{n} \sum_i x_i^{d,aggr}$, $y^{d,aggr} = \frac{1}{n} \sum_i y_i^{d,aggr}$. The generation of test output given the test input will be conditioned on these aggregated context pairs.
# 
# In the other path, input/output representation across different demo pairs are concatenated ($x^{d,cat}$, $y^{d,cat}$).
# 
# ### - Predicting test outputs conditioned on the context pairs
# 
# The test input ($x^{t}$) passes through the same initial encoding as demo inputs to produce $x^{t,enc}$. The test output ($y^{t}$) is initialized as blank with added positional encoding to produce $y^{t,enc}$. To condition on the context pairs, the aggregated demo input/output representations $x^{d,aggr}$, $y^{d,aggr}$ are added to test input/output encodings. The conditioning can also be made probablistic as in neural processes. 
# 
# Finaly the encoded test input, the concatenated demo input/output representations, and the encoded test output goes through a series of configurable self- and cross-attention blocks to generate the output image. 
# 

# In[ ]:


import torch
import torch.nn as nn
import time
import os
import sys
import argparse
import logging
import random
import numpy as np
import pickle
import collections
import pandas as pd
from datetime import datetime, timedelta
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter

import plot_utils
import data_utils


# In[ ]:


TEST_DIR = "../input/abstraction-and-reasoning-challenge/test/"
EVAL_EVERY = None  # set eval_every to None

PRESUBMIT = False
fake_test_files = os.listdir(TEST_DIR)
if "0c9aba6e.json" in fake_test_files:
    PRESUBMIT = True
    TEST_DIR = "../input/arc-eval-100/eval_100/"
    EVAL_EVERY = 20


# In[ ]:


KAGGLE_KERNEL = True
INIT_MODEL = "../input/arc-pretrained-13990/pretrained_13990.pt"
TIME_LIMIT = 110  # minutes
CONFIG = {
    "init_model": INIT_MODEL,  # pretrained model path
    "n_ensemble": 3,  # number of final predictions
    "name": "Kaggle",  # extra name
    "message": "",  # notes
    "n_epochs": 3000,  # number of epochs (if within time limit)
    "time_limit": TIME_LIMIT,  # time limit before stop (in minutes)
    "lr": 0.0015,  # learning rate
    "half_lr_after": None,  # reduce the lr by half after ? epochs
    "optim_update_every": 150,  # accumulate gradient for ? iterations before optimizer update
    "eval_every": EVAL_EVERY,  # eval every ? epochs
    "save_every": 10,  # save every ? epochs
    "print_every": 20,  # eval every ? epochs
    "n_checkpoints_kept": 5,  # how many checkpoints to keep
    "resume": "",  # checkpoint path to resume from
    "gpu": 0,  # which gpu
    "aux_ratio": 0.3  # auxiliary loss ratio
}
args = SimpleNamespace(**CONFIG)  # to mimic argparse api


# In[ ]:


# logging
log = logging.getLogger()
log.setLevel(logging.INFO)
if (KAGGLE_KERNEL and len(log.handlers) <=2) or not log.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    log.addHandler(handler)

START_TIME = datetime.now()
log.info("Current time: " + START_TIME.strftime("%Y%m%d %H:%M"))
DEVICE = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
log.info("Using device {}".format(DEVICE))


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import math

import numpy as np


def build_model(device="cuda"):
    d_enc, d_ff = 128, 512
    conv_encoder = None
    conv_encoder = ConvImageEncoder(d_enc, n_blocks=1)
    demo_encoder = SCAB_Encoder_Batch(
        n_self_attn=1, n_cross_attn=1, d_enc=d_enc, n_heads=4, d_ff=d_ff, drop=0.1)
    test_decoder = SCAB_Decoder(
        layers=["xy", "dy", "xy"], d_enc=d_enc, n_heads=4, d_ff=d_ff, drop=0.1)
    model = ArcTransformer_MultiPath_Wrapper(
        d_enc=d_enc, demo_encoder=demo_encoder, test_decoder=test_decoder, conv_encoder=conv_encoder,
        aggr_x=True, aggr_y=True, aggr_x_cat=False, aggr_y_cat=False, aggr_shrink=3, device=device).to(device)
    return model


class ArcTransformer_MultiPath_Wrapper(nn.Module):
    def __init__(self, d_enc, demo_encoder, test_decoder, conv_encoder=None, n_colors=10,
                 aggr_x=True, aggr_y=True, aggr_x_cat=False, aggr_y_cat=False, aggr_shrink=1, device="cuda"):
        super().__init__()
        self.d_enc = d_enc
        self.aggr_x = aggr_x
        self.aggr_y = aggr_y
        self.aggr_x_cat = aggr_x_cat
        self.aggr_y_cat = aggr_y_cat
        self.aggr_shrink = aggr_shrink
        self.positional_encoder = GridPositionalEncoderWithCache(
            d_enc=d_enc, max_img_shape=(30, 30), device=device)
        
        self.conv_encoder = conv_encoder

        self.enc1 = nn.Sequential(  # (w_in, h_in)
            UnsqueezeBatch(),  # (1, w_in, h_in)
            nn.Embedding(n_colors, d_enc),  # (1, w_in, h_in, d_embed)
        )

        self.enc2 = nn.Sequential(
            self.positional_encoder,  # (1, w_in, h_in, d_embed)
            nn.Flatten(start_dim=-3, end_dim=-2),  # (1, n_nodes, d_embed)
        )
        self.enc_y_test = nn.Sequential(  # (w_in, h_in, d_embed)
            UnsqueezeBatch(),  # (1, w_in, h_in, d_embed)
            # no embedding
            self.positional_encoder,  # (1, w_in, h_in, d_embed)
            nn.Flatten(start_dim=-3, end_dim=-2),  # (1, n_nodes, d_embed)
        )

        self.demo_encoder = demo_encoder
        self.test_decoder = test_decoder

        self.fc_out = nn.Linear(d_enc, n_colors)

    def forward(self, x_demos_raw, y_demos_raw, x_test, y_test_shape):
        n_demos = len(x_demos_raw)

        x_demos, y_demos = [None] * n_demos, [None] * n_demos
        for i in range(n_demos):
            x_demos[i] = self.enc1(x_demos_raw[i])  # (1, w_in, h_in, d_enc)
            y_demos[i] = self.enc1(y_demos_raw[i])  # (1, w_in, h_in, d_enc)
        x_test = self.enc1(x_test)
        
        if self.conv_encoder is not None:
            all_demos = batch_2d_layer(self.conv_encoder, x_demos + y_demos + [x_test])
            x_demos, y_demos, x_test = all_demos[:len(all_demos)//2], all_demos[len(all_demos)//2:-1], all_demos[-1]

        for i in range(n_demos):
            x_demos[i] = self.enc2(x_demos[i])  # (1, x_nodes, d_enc)
            y_demos[i] = self.enc2(y_demos[i])  # (1, y_nodes, d_enc)
        x_test = self.enc2(x_test)

        y_test = torch.zeros((*y_test_shape, self.d_enc),
                             dtype=torch.float, device=x_test.device)
        
        
        y_test = self.enc_y_test(y_test)  # (1, n_nodes, d_enc)

        # transformer layers
        y_aggr, (x_demos_cat, y_demos_cat, x_aggr) =             self.demo_encoder(x_demos, y_demos)
        
        if self.aggr_x:
            if self.aggr_x_cat:
                x_test = torch.cat([x_test, x_aggr.repeat(1, x_test.shape[1], 1)], axis=-1)
            else:
                x_test += x_aggr / self.aggr_shrink
        
        if self.aggr_y:
            if self.aggr_y_cat:
                y_test = torch.cat([y_test, y_aggr.repeat(1, y_test.shape[1], 1)], axis=-1)
            else:
                y_test += y_aggr / self.aggr_shrink
        
        y_test, x_test, auxiliary =             self.test_decoder(x_demos_cat, y_demos_cat, x_test, y_test)

        y_test = self.fc_out(y_test)
        y_test = F.log_softmax(y_test, dim=-1)  # (1, n_nodes, n_colors)
        y_test = y_test.view((1, *y_test_shape, y_test.shape[-1]))

        aux_out = []
        for y in auxiliary:
            aux = self.fc_out(y)
            aux = F.log_softmax(aux, dim=-1)  # (1, n_nodes, n_colors)
            aux = aux.view((1, *y_test_shape, aux.shape[-1]))
            aux_out.append(aux)

        return y_test, (x_demos_cat, y_demos_cat, x_test), aux_out  # (1, w_out, h_out, d_out)


class SCAB_Encoder_Batch(nn.Module):
    def __init__(self, n_self_attn=1, n_cross_attn=1, d_enc=128, n_heads=4, d_ff=None, drop=0, ln=True):
        super().__init__()
        self.n_self_attn = n_self_attn
        self.n_cross_attn = n_cross_attn
        # aggr
        self.aggr_x = nn.Sequential(
            nn.Linear(d_enc, d_enc),
            nn.ReLU(),
            nn.Linear(d_enc, d_enc))
        self.aggr_y = nn.Sequential(
            nn.Linear(d_enc, d_enc),
            nn.ReLU(),
            nn.Linear(d_enc, d_enc))

        # enc demo inputs
        self.demo_self_enc = nn.ModuleList(
            [SAB(d_enc, d_enc, num_heads=n_heads, dim_F=d_ff, drop=drop, ln=ln) \
                for _ in range(n_self_attn)])
        self.demo_yx_enc = nn.ModuleList(
            [SCAB(d_enc, d_enc, d_enc, num_heads=n_heads, dim_F=d_ff, drop=drop, ln=ln) \
                for _ in range(n_cross_attn)])
        self.demo_xy_enc = nn.ModuleList(
            [SCAB(d_enc, d_enc, d_enc, num_heads=n_heads, dim_F=d_ff, drop=drop, ln=ln) \
                for _ in range(n_cross_attn)])

    def forward(self, x_demos, y_demos):
        # encode demos
        n_demos = len(x_demos)

        for i in range(self.n_self_attn):
            all_demos = batch_layer(self.demo_self_enc[i], x_demos + y_demos)
            x_demos, y_demos = all_demos[:len(all_demos)//2], all_demos[len(all_demos)//2:]
        
        for i in range(self.n_cross_attn):
            x_demos, y_demos =                 batch_cross_attn(self.demo_xy_enc[i], x_demos, y_demos),                 batch_cross_attn(self.demo_yx_enc[i], y_demos, x_demos)

        inds_x = np.cumsum([0] + [d.shape[1] for d in x_demos]).tolist()  # [0, 3, 7, ...]
        inds_y = np.cumsum([0] + [d.shape[1] for d in y_demos]).tolist()

        x_demos_cat = torch.cat(x_demos, axis=1)  # (1, n_total, d_enc)
        y_demos_cat = torch.cat(y_demos, axis=1)

        x_demos_aggr = x_demos_cat + self.aggr_x(x_demos_cat)
        y_demos_aggr = y_demos_cat + self.aggr_y(y_demos_cat)

        x_aggr = 0
        for i in range(n_demos):
            x_aggr += x_demos_aggr[:, inds_x[i]:inds_x[i+1]].mean(1) / n_demos
        y_aggr = 0
        for i in range(n_demos):
            y_aggr += y_demos_aggr[:, inds_y[i]:inds_y[i+1]].mean(1) / n_demos

        return y_aggr, (x_demos_cat, y_demos_cat, x_aggr)


class SCAB_Decoder(nn.Module):
    """
    layers: 'x' means self-attn(x_test), 'y' means cross-attn(y_test, x_test),
        'd' -- cross-attn(ytest or x_test, y_demos_cat or x_demos_cat)
    """

    def __init__(self, layers=["x", "xy", "dy", "dy"], d_enc=128, n_heads=4, d_ff=None, drop=0,
                 ln=True):
        super().__init__()
        assert("y" in layers[-1])

        self.layers = layers
        self.n_x_enc = len([s for s in layers if "x" in s])
        self.n_yx_dec = len([s for s in layers if "y" in s])
        self.n_demo_dec = len([s for s in layers if "d" in s])

        # enc x_test
        self.x_enc = nn.ModuleList(
            [SAB(d_enc, d_enc, num_heads=n_heads, dim_F=d_ff, drop=drop, ln=ln) \
                for _ in range(self.n_x_enc)])
        
        # dec y_test
        self.yx_dec = nn.ModuleList(
            [SCAB(d_enc, d_enc, d_enc, num_heads=n_heads, dim_F=d_ff, drop=drop, ln=ln) \
                for _ in range(self.n_yx_dec)])
        
        self.x_vert_dec = nn.ModuleList(
            [SCAB(d_enc, d_enc, d_enc, num_heads=n_heads, dim_F=d_ff, drop=drop, ln=ln) \
                for _ in range(self.n_demo_dec)])
        self.y_vert_dec = nn.ModuleList(
            [SCAB(d_enc, d_enc, d_enc, num_heads=n_heads, dim_F=d_ff, drop=drop, ln=ln) \
                for _ in range(self.n_demo_dec)])

    def forward(self, x_demos_cat, y_demos_cat, x_test, y_test):

        auxiliary = []
        xi, di, yi = 0, 0, 0
        for i, layer in enumerate(self.layers):
            if "x" in layer:
                x_test = self.x_enc[xi](x_test)
                xi += 1
            if "d" in layer:
                x_test = self.x_vert_dec[di](x_test, x_demos_cat)
                y_test = self.y_vert_dec[di](y_test, y_demos_cat)
                di += 1
            if "y" in layer:
                y_test = self.yx_dec[yi](y_test, x_test)
                yi += 1
                if i != len(self.layers) - 1:
                    auxiliary.append(y_test)

        return y_test, x_test, auxiliary


def batch_layer(function, tensors):
    """
    function: a neural network that takes a signle tensor
    tensors: list of tensors of shape [1, n_nodes, n_dims], each tensor can have 
        different n_nodes, but all need to have the same n_dims
    """
    n_tensors = len(tensors)
    lengths = [x.shape[1] for x in tensors]
    n_dims = tensors[0].shape[2]
    batch = torch.zeros((n_tensors, max(lengths), n_dims), device=tensors[0].device)
    mask = torch.zeros((n_tensors, max(lengths)), dtype=torch.bool, device=tensors[0].device)
    for i, tensor in enumerate(tensors):
        batch[i, 0:lengths[i]] = tensor
        mask[i, 0:lengths[i]] = 1
    batch = function(batch, mask=mask)

    out = [None for _ in range(len(tensors))]
    for i in range(len(tensors)):
        out[i] = batch[i, 0:lengths[i]].unsqueeze(0)
    return out


def batch_2d_layer(function, tensors):
    """
    function: a neural network that takes a signle tensor
    tensors: list of tensors of shape [1, n_nodes, n_dims], each tensor can have 
        different n_nodes, but all need to have the same n_dims
    """
    n_tensors = len(tensors)
    widths, heights = [x.shape[1] for x in tensors], [x.shape[2] for x in tensors]
    n_dims = tensors[0].shape[3]
    batch = torch.zeros((n_tensors, max(widths), max(heights), n_dims), device=tensors[0].device)
    mask = torch.zeros((n_tensors, max(widths), max(heights)), dtype=torch.bool, device=tensors[0].device)
    for i, tensor in enumerate(tensors):
        batch[i, 0:widths[i], 0:heights[i]] = tensor
        mask[i, 0:widths[i], 0:heights[i]] = 1
    batch = function(batch, mask=mask)

    out = [None for _ in range(len(tensors))]
    for i in range(len(tensors)):
        out[i] = batch[i, 0:widths[i], 0:heights[i]].unsqueeze(0)
    return out


def batch_cross_attn(function, this, other):
    """
    function: a neural network that takes a two tensors and returns a single tensor
    this: list of tensors of shape [1, n_nodes, n_dims], each tensor can have
        different n_nodes, but all need to have the same n_dims.
    other: must be the same length as this
    """
    assert(len(this) == len(other))
    n_tensors = len(this)

    len_this = [x.shape[1] for x in this]
    n_dims = this[0].shape[2]
    batch_this = torch.zeros((n_tensors, max(len_this), n_dims), device=this[0].device)
    mask_this = torch.zeros((n_tensors, max(len_this)), dtype=torch.bool, device=this[0].device)
    for i, tensor in enumerate(this):
        batch_this[i, 0:len_this[i]] = tensor
        mask_this[i, 0:len_this[i]] = 1

    len_other = [x.shape[1] for x in other]
    n_dims = other[0].shape[2]
    batch_other = torch.zeros((n_tensors, max(len_other), n_dims), device=other[0].device)
    mask_other = torch.zeros((n_tensors, max(len_other)), dtype=torch.bool, device=other[0].device)
    for i, tensor in enumerate(other):
        batch_other[i, 0:len_other[i]] = tensor
        mask_other[i, 0:len_other[i]] = 1

    batch_this = function(batch_this, batch_other, mask_this, mask_other)
    # should not mutates the input tensor array
    out = [None for _ in range(len(this))]
    for i in range(len(this)):
        out[i] = batch_this[i, 0:len_this[i]].unsqueeze(0)
    return out


class MAB(nn.Module):
    # adapted from https://github.com/juho-lee/set_transformer
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, dim_F=None, drop=0, ln=False, ff=True):
        super(MAB, self).__init__()
        self.dim_Q = dim_Q
        self.dim_V = dim_V
        if dim_F is None:
            dim_F = dim_V * 4
        self.num_heads = num_heads
        self.feed_forward = ff
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o1 = nn.Linear(dim_V, dim_F)
        self.fc_o2 = nn.Linear(dim_F, dim_V)
        self.dropout1 = nn.Dropout(drop)
        self.dropout2 = nn.Dropout(drop)

    def forward(self, Q, K, mask=None):
        Q_ori = Q
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)  # (batch*n_heads, n, dim_split)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        scores = Q_.bmm(K_.transpose(1, 2))/math.sqrt(self.dim_V)  # (batch, n_q, n_k)
        if mask is not None:  # mask the scores along the K axis
            mask = mask.repeat(self.num_heads, 1).unsqueeze(1)  # (batch*n_heads, 1, n_K)
            # scores = scores.masked_fill(mask == 0, -1e9)
            scores.masked_fill_(mask == 0, -1e9)

        A = torch.softmax(scores, 2)
        # original MAB implementation
        # O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        # changed to real residual connection (next two lines)
        O = torch.cat(A.bmm(V_).split(Q.size(0), 0), 2)
        if self.dim_Q == self.dim_V:
            O = Q_ori + self.dropout1(O)  # this requires dim_Q == dim_V
        else:
            O = Q + self.dropout1(O)

        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        if self.feed_forward:
            O = O + self.dropout2(self.fc_o2(F.relu(self.fc_o1(O))))
            O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, dim_F=None, drop=0, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, dim_F, drop=drop, ln=ln)

    def forward(self, X, mask=None):
        return self.mab(X, X, mask=mask)


class SCAB(nn.Module):
    def __init__(self, dim_q, dim_k, dim_out, num_heads, dim_F=None, drop=0, ln=False):
        super().__init__()
        self.mab0 = MAB(dim_q, dim_q, dim_out, num_heads, dim_F, drop=drop, ln=ln, ff=False)
        self.mab1 = MAB(dim_out, dim_k, dim_out, num_heads, dim_F, drop=drop, ln=ln)

    def forward(self, Xq, Xk, mask_q=None, mask_k=None):
        return self.mab1(self.mab0(Xq, Xq, mask=mask_q), Xk, mask=mask_k)


class UnsqueezeBatch(nn.Module):
    def forward(self, X):
        return X.unsqueeze(0)


class GridPositionalEncoderWithCache(nn.Module):
    # adapted from https://github.com/sahajgarg/image_transformer
    def __init__(self, d_enc, max_img_shape, min_timescale=1.0, max_timescale=1.0e4,
                 num_dims=2, device="cpu"):
        super().__init__()
        self.d_enc = d_enc
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.num_dims = num_dims
        self.full_size_signal = self.positional_signal(max_img_shape, device)

    def positional_signal(self, shape, device="cpu"):
        # X: (batch, x, y, hidden)
        # return: add in the range of (-1, 1)
        num_timescales = self.d_enc // (self.num_dims * 2)
        log_timescale_increment = np.log(
            self.max_timescale / self.min_timescale) / (num_timescales - 1)
        inv_timescales = self.min_timescale *             torch.exp((torch.arange(num_timescales).float()
                       * -log_timescale_increment))
        inv_timescales = inv_timescales.to(device)
        total_signal = torch.zeros((1, shape[0], shape[1], self.d_enc), device=device)
        for dim in range(self.num_dims):
            length = shape[dim]
            position = torch.arange(length).float().to(device)
            scaled_time = position.view(-1, 1) * inv_timescales.view(1, -1)
            signal = torch.cat(
                [torch.sin(scaled_time), torch.cos(scaled_time)], 1)
            prepad = dim * 2 * num_timescales
            postpad = self.d_enc - (dim + 1) * 2 * num_timescales
            signal = F.pad(signal, (prepad, postpad))
            for _ in range(1 + dim):
                signal = signal.unsqueeze(0)
            for _ in range(self.num_dims - 1 - dim):
                signal = signal.unsqueeze(-2)
            # X += signal
            total_signal += signal
        return total_signal

    def forward(self, X):
        # X: (batch, width, height, d_enc)
        X = X + self.full_size_signal[:, :X.shape[1], :X.shape[2]]
        return X


class ConvImageEncoder(nn.Module):
    def __init__(self, d_enc, n_blocks=1):
        super(ConvImageEncoder, self).__init__()
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList([BasicBlock(d_enc, d_enc, stride=1) for _ in range(n_blocks)])

    def forward(self, x, mask=None):
        out = x.permute(0, 3, 1, 2).contiguous()
        for i in range(self.n_blocks):
            out = self.blocks[i](out, mask=mask)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out


class BasicBlock(nn.Module):
    # https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(self.expansion*planes)
            )

    def forward(self, x, mask=None):
        out = F.relu(self.bn1(self.conv1(x)))
        if mask is not None:
            out.masked_fill_(mask.unsqueeze(1) == 0, -1e9)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if mask is not None:
            out.masked_fill_(mask.unsqueeze(1) == 0, -1e9)
        out = F.relu(out)
        return out


# In[ ]:


def train(training_set, test_set, model_id):
    # build model (must be the same model as the pretrained checkpoint)
    model = build_model(DEVICE)

    training_ids = sorted(list(training_set.keys()))
    test_ids = sorted(list(test_set.keys()))
    # initialize stats
    train_losses = []
    eval_losses = []
    train_top1 = []
    eval_top1 = []
    train_soft_acc = []
    eval_soft_acc = []
    min_eval_loss = np.inf  # not a good indicator of improvement
    max_eval_soft_acc = 0
    max_eval_soft_acc_fname = ""
    fname_queue = collections.deque()

    # training parameters
    lr = args.lr
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=0)
    criterion = nn.NLLLoss()

    epoch_start = 1
    n_iter = 0

    if not args.resume:
        if args.init_model:
            # start from a pretrained checkpoint
            checkpoint = torch.load(args.init_model, map_location='cuda:{}'.format(args.gpu))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(DEVICE)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            model_id += "e{}".format(checkpoint['epoch'])
        else:
            # start from scratch
            log.warning("Training from scratch.")
            model_id += "e0"
    else:
        # resume from partially trainded model
        model_checkpoint = args.resume
        checkpoint = torch.load(model_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        epoch_start = checkpoint['epoch'] + 1
        n_iter = checkpoint['epoch'] * len(training_ids)
        with open(model_checkpoint[:-3] + "_stats.pkl", 'rb') as f:
            train_losses, eval_losses, train_top1, eval_top1 = pickle.load(f)
        model_id = os.path.dirname(model_checkpoint).split('/')[-1]
    
    # save checkpoints in model_dir, save stats in log_dir
    model_dir = log_dir = model_id + "/"
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    if args.message:
        with open(os.path.join(log_dir, "README_MICROTRAIN.md"), "w") as f:
            f.write(args.message)
    writer = SummaryWriter(log_dir=log_dir)
    
    optimizer.zero_grad()

    # run the training loop
    for epoch in range(epoch_start, args.n_epochs + 1):
        current_time = datetime.now()
        since_start = (current_time - START_TIME).total_seconds() / 60  # in minutes
        if since_start >= args.time_limit:
            log.info("Stopping training at {} epochs.".format(epoch))
            log.info("Current time: {}, start time: {}".format(
                current_time.strftime("%Y%m%d %H:%M"), START_TIME.strftime("%Y%m%d %H:%M")))
            break

        if args.half_lr_after == epoch:
            lr /= 2.0
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t_start = time.time()
        shuffled_ids = training_ids[:]
        random.shuffle(shuffled_ids)

        model.train()
        losses_epoch = []
        top1_epoch = 0
        soft_acc_epoch = 0
        aux_losses_epoch = []

        for train_id in shuffled_ids:
            n_iter += 1
            data = training_set[train_id]

            # data augmentation
            data = data_utils.shuffle_demo_test(data, demo_only=True)  # only using demo data
            # add flip/roate augmentation
            data = data_utils.random_flip(data)
            data = data_utils.random_transpose(data)
            data, perm_colors = data_utils.permute_colors(data, device=DEVICE)

            x_demos, y_demos = [], []
            for demo in data['train']:
                x_demos.append(demo['input'])
                y_demos.append(demo['output'])

            test_id = random.randrange(len(data['test']))
            test_pair = data['test'][test_id]

            x_test, y_test = test_pair['input'], test_pair['output']

            y_test_shape = y_test.shape  # provide known shape in training

            y_prob, _, auxiliary = model(x_demos, y_demos, x_test, y_test_shape)
            assert(y_prob.argmax(-1).squeeze(0).shape == y_test_shape)

            loss = criterion(y_prob.permute(0, 3, 1, 2), y_test.unsqueeze(0))
            y_pred = y_prob.argmax(-1).squeeze(0)
            top1_epoch += torch.equal(y_pred, y_test)
            soft_acc_epoch += torch.mean((y_pred == y_test).float()).item()

            if args.aux_ratio > 0:
                # if use auxiliary loss
                if auxiliary is None:
                    raise AssertionError("aux ratio > 0 but no Aux loss detected.")
                total_loss = 0
                for aux in auxiliary:
                    total_loss += args.aux_ratio * criterion(aux.permute(0, 3, 1, 2), y_test.unsqueeze(0))
                total_loss += loss
                total_loss.backward()
                aux_losses_epoch.append(total_loss.item())
            else:
                # normal loss
                loss.backward()

            # batch optimizer update (TODO: parallelize batch update)
            if n_iter % args.optim_update_every == 0:
                optimizer.step()
                optimizer.zero_grad()

            losses_epoch.append(loss.item())

        # stats
        losses_epoch = np.mean(losses_epoch)
        top1_epoch = top1_epoch / float(len(training_ids))
        soft_acc_epoch = soft_acc_epoch / float(len(training_ids))
        train_losses.append(losses_epoch)
        train_top1.append(top1_epoch)
        train_soft_acc.append(soft_acc_epoch)
        time_per_epoch = (time.time() - t_start)
        time_per_iter = time_per_epoch / len(shuffled_ids)
        time_now = datetime.now().strftime("%m%d%H%M")
        log_str = "{} ep:{} it:{} NLL:{:.3f} soft acc: {:.3f} top1:{:.4f} t/ep:{:.1f} t/it:{:.3f} lr:{}".format(
            time_now, epoch, n_iter, losses_epoch, soft_acc_epoch, top1_epoch, time_per_epoch, time_per_iter, lr)
        if args.aux_ratio:
            log_str += ", NLL+Aux:{:.3f}".format(np.mean(aux_losses_epoch))
        if args.print_every and epoch % args.print_every == 0:
            log.info(log_str)
        writer.add_scalar('Loss/Train', losses_epoch, epoch)
        writer.add_scalar('Accuracy_top1/Train', top1_epoch, epoch)
        writer.add_scalar('Soft_Accuracy/Train', soft_acc_epoch, epoch)

        # Evaluation
        if args.eval_every and epoch % args.eval_every == 0:
            torch.cuda.empty_cache()
            model.eval()

            eval_loss = []
            top1_epoch = 0
            soft_acc_epoch = 0
            for test_id in test_ids:
                data = test_set[test_id]
                x_demos, y_demos = [], []
                for demo in data['train']:
                    x_demos.append(demo['input'])
                    y_demos.append(demo['output'])

                test_id = 0
                test_pair = data['test'][test_id]

                x_test, y_test = test_pair['input'], test_pair['output']
                
                y_test_shape = y_test.shape  # provide known shape during training

                y_prob, _, _ = model(x_demos, y_demos, x_test, y_test_shape)
                assert(y_prob.argmax(-1).squeeze(0).shape == y_test_shape)

                loss = criterion(y_prob.permute(0, 3, 1, 2), y_test.unsqueeze(0))
                y_pred = y_prob.argmax(-1).squeeze(0)
                top1_epoch += torch.equal(y_pred, y_test)
                soft_acc_epoch += torch.mean((y_pred == y_test).float()).item()

                eval_loss.append(loss.item())

            eval_loss = np.mean(eval_loss)
            top1_epoch = top1_epoch / float(len(test_ids))
            soft_acc_epoch = soft_acc_epoch / float(len(test_ids))
            eval_losses.append((epoch, eval_loss))
            eval_top1.append((epoch, top1_epoch))
            eval_soft_acc.append((epoch, soft_acc_epoch))
            log.info("{} Eval, NLL:{:.3f} soft acc: {:.3f} top1:{:.4f}".format(model_id, eval_loss, soft_acc_epoch, top1_epoch))
            writer.add_scalar('Loss/Validation', eval_loss, epoch)
            writer.add_scalar('Accuracy_top1/Validation', top1_epoch, epoch)
            writer.add_scalar('Soft_Accuracy/Validation', soft_acc_epoch, epoch)

        if args.save_every and epoch % args.save_every == 0:
            # remove previous checkpoints
            if len(fname_queue) >= args.n_checkpoints_kept:
                fname = fname_queue.popleft()
                if os.path.isfile(fname):
                    os.remove(fname)
                pickle_fname = fname.strip(".pt") + "_stats.pkl"
                if os.path.isfile(pickle_fname):
                    os.remove(pickle_fname)
            
            # save model checkpoint
            fname = os.path.join(model_dir, model_id + '_' + str(epoch) + '.pt')
            torch_save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(torch_save_dict, fname)
            fname_queue.append(fname)

            # save stats in pickle file
            pickle_fname = '{}_stats.pkl'.format(fname.strip(".pt"))
            with open(pickle_fname, 'wb') as f:
                pickle.dump([train_losses, eval_losses, train_top1, eval_top1], f)
            
            # save best validation checkpoint
            if args.eval_every and soft_acc_epoch > max_eval_soft_acc:
                if os.path.isfile(max_eval_soft_acc_fname):
                    os.remove(max_eval_soft_acc_fname)
                max_eval_soft_acc_fname = os.path.join(model_dir, '_best_eval_' + str(epoch) + '.pt')
                torch.save(torch_save_dict, max_eval_soft_acc_fname)
                max_eval_soft_acc = soft_acc_epoch

    return model, (fname_queue, max_eval_soft_acc_fname)


# In[ ]:


def predict(model, test_set, plot=False):
    model.eval()
    test_ids = list(test_set.keys())
    res = []
    for test_id in test_ids:
        data = test_set[test_id]

        x_demos, y_demos = [], []
        for demo in data['train']:
            x_demos.append(demo['input'])
            y_demos.append(demo['output'])

        # test_output = data_utils.create_dummy_test_output(test_data)
        for i, test_pair in enumerate(data['test']):
            x_test = test_pair['input']
            # TODO: add better shape prediction
            y_test_shape = data_utils.predict_shape(data, x_test)
            y_prob, _, _ = model(x_demos, y_demos, x_test, y_test_shape)
            y_pred1 = y_prob.argmax(-1).squeeze(0)
            test_pair['pred'] = y_pred1.detach().cpu()
            
            y_pred1 = data_utils.flattener(y_pred1.cpu().numpy().tolist())
            
            output_string = y_pred1
            # top3 = [y_pred1, y_pred1, y_pred1]  # TODO: get the top 3
            # output_string = ' '.join(top3)
            output_id = test_id + '_' + str(i)
            res.append({'output_id': output_id, 'output': output_string})
        
        if plot:
            plot_utils.plot_task(data)
            
    return pd.DataFrame(res, columns=['output_id', 'output'])


# In[ ]:


model_name = "Microtrain"
model_id = model_name + datetime.now().strftime("x%Y%m%dx%H%M")
if args.name:
    model_id += "x" + args.name

# load test data
test_raw = data_utils.load_dataset(TEST_DIR)
test_set = data_utils.dataset_to_tensor(test_raw, device=DEVICE)
# load micro-training data (without test input/output)
training_set = data_utils.dataset_to_tensor(
    test_raw, device=DEVICE, include_test=False)

log.info("Loaded {} test examples.".format(len(test_set)))

model, (fname_queue, _) = train(training_set, test_set, model_id)


# In[ ]:


torch.cuda.empty_cache()
res = predict(model, test_set, plot=True)

if args.n_ensemble > 1:
    res = [res]

    # other single checkpoints
    n_models = args.n_ensemble - 1
    checkpoint_fnames = list(fname_queue)[-(n_models+1): -1]
    for model_checkpoint in checkpoint_fnames:
        del model
        torch.cuda.empty_cache()
        model = build_model(DEVICE)
        checkpoint = torch.load(model_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        res.append(predict(model, test_set))
    res = data_utils.concat_results(res)
print(res)
res.to_csv('submission.csv', index=False)
log.info("Done. {}".format(datetime.now().strftime("%Y%m%d %H:%M")))


# In[ ]:


if PRESUBMIT:
    import shutil
    shutil.copyfile('submission.csv', 'val_submission.csv')
    sample = pd.read_csv("../input/abstraction-and-reasoning-challenge/sample_submission.csv")
    sample.to_csv('submission.csv', index=False)


# In[ ]:




