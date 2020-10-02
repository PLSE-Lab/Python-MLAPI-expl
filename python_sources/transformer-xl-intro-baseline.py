#!/usr/bin/env python
# coding: utf-8

# A lot of people (i.e kkiller, KeepLearning etc.) have wanted this kernel so here it is.

# Most ideas are taken from the NVIDIA repo (https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/mem_transformer.py). Credit to them.

#  # Have to give credit to Nvidia and their wonderful GitHub repo.
#  
#  This is a hybridized Transformer-XL which also uses:
#  * CBR Blocks
#  * LSTM-base FFNs
#  * Adaptive Embeddings
# 
#  
#  This has adapted their work into PyTorch. I prefer TF, but PyTorch has the benefit of Open.Ai and their huge ecosystem (Pyro.ai etc).
#  The TFXL was previously written in TF2.1, but I converted it into PyTorch. Why?
#  
#  ### The main reason?
#  
#  **MEGATRON-LM**
#  
#  The monstrous state-of-the-art transformer is the main reason why I am working with PyTorch. I myself only used the TFXL until now.

# My CSV files are in [this dataset](https://www.kaggle.com/nxrprime/preprocessing-for-m5) as well as the official M5 dataset.

# Unfortunately, the Kaggle kernels environment has very less memory. I cannot run the full NN on Kaggle kernels, so  it has been quicksaved.

# # Why such a big model?

# This model is **TITANIC.** It is very, very big and has so many components like RNNs, positional encodings etc. But why? Why should I create such a monstrosity of titanic proportions and let it tango with the M5 dataset (which is by no means small)? The answer is simple: **this is to beat N-BEATS.** (despite being a long, long way off from that goal of triumph over Yoshua Bengio).
# 
# Mammoth models often tend to have **ONE bad egg in them**, so over here I'm guessing I'll need to fix the positional encoding. It is literally impossible to perform any sort of increase in the parameters beyond the optimum without a titanic amount of RAM. 

# # A Full Training Pipeline
# 
# ### Link: https://github.com/Trigram19/m5-python-starter
# 
# This pipeline by no means is my complete version: it is a *distilled version.*

# # Imports

# Just importing the necessary libraries.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc; import os
import torch
from torch.nn import *
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os
from tqdm.notebook import tqdm


# In[ ]:


import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


# In[ ]:


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


# ## Definition of functions

# ### Megatron-LM GPT2 Layer (for later)

# This is a huge and state of the art Attention module. We will be using many besides this as well.

# ### Feed-forwards
# 
# We will implement these in our decoders.

# In[ ]:


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None])                     .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError

class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output

class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen-r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen-r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias[None]                                   # qlen x bsz x n_head x d_head

        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))                  # qlen x klen x bsz x n_head
        D_ = r_bias[None, :, None]                                              # 1    x klen x 1   x n_head
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


# In[ ]:


class PositionwiseLSTMFF(torch.nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseLSTMFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), 
            nn.LSTMCell(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


# In[ ]:


class PositionwiseGRUFF(torch.nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseGRUFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), 
            nn.GRUCell(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


# In[ ]:


# Create Attention mask
def create_mask(qlen, mlen, dtype=torch.float32, same_length=False):
      """Creates attention mask when single-side context allowed only."""
      attn_mask = torch.ones([qlen, qlen], dtype=dtype)
      mask_u = torch.triu(attn_mask, 0, -1)
      mask_dia = torch.triu(attn_mask, 0, 0)
      attn_mask_pad = torch.zeros([qlen, mlen], dtype=dtype)
      ret = torch.concat([attn_mask_pad, mask_u - mask_dia], 1)
      if same_length:
        mask_l = torch.triu(attn_mask, -1, 0)
        ret = torch.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)

      return ret


# ### Embeddings and attention

# In[ ]:


class PositionalEmbedding(Module):
  def __init__(self, dim, **kwargs):
    super(PositionalEmbedding, self).__init__(**kwargs)
    self.dim = dim

    """Constructs inversed frequency vector for positional embedding layer."""
    self.inv_freq = 1.0 / (10000.0**(torch.range(0, 19380, 10.0) / self.dim))

  def forward(self, pos_seq, batch_size):
    """Implements call() for the layer."""
    sinusoid_inp = torch.einsum('i,d->id', pos_seq, self.inv_freq)
    pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], -1)
    pos_emb = pos_emb[:, None, :]

    if batch_size is not None:
      pos_emb = tile(pos_emb, 2, self.dim)

    return pos_emb


# In[ ]:


## Took this from the TF GitHub repo and translated into Pytorch ##
class RelativeAttention(Module):
  """Core calculations for relative attention."""

  def __init__(self, dropout_att, scale):
    super(RelativeAttention, self).__init__()
    self.scale = scale
    self.dropout_att = dropout_att

  def build(self, unused_input_shapes):
    self.attention_probs_dropout = torch.nn.Dropout(
        p=self.dropout_att)

    super(RelativeAttention, self).build(unused_input_shapes)

  def call(self, q_head, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
           r_w_bias, r_r_bias, r_s_bias, attn_mask):
    # content based attention score
    ac = torch.einsum('ibnd,jbnd->ijbn', q_head + r_w_bias, k_head_h)

    # position based attention score
    bd = torch.einsum('ibnd,jbnd->ijbn', q_head + r_r_bias, k_head_r)
    bd = rel_shift(bd, klen=tf.shape(ac)[1])

    # segment-based attention score
    if seg_mat is None:
      ef = 0
    else:
      ef = torch.einsum('ibnd,snd->isbn', q_head + r_s_bias, seg_embed)
      tgt_shape = torch.shape(bd)
      ef = torch.where(
          torch.Tensor(np.broadcast_to(torch.expand_dims(seg_mat, 3), tgt_shape)),
          torch.Tensor(np.broadcast_to(ef[:, 1:, :, :], tgt_shape)),
          torch.Tensor(np.broadcast_to(ef[:, :1, :, :], tgt_shape)))

    # merges attention scores and performs masking
    attn_score = (ac + bd + ef) * self.scale
    if attn_mask is not None:
      attn_score = attn_score - 1e30 * attn_mask

    # attention probability
    attn_prob = functional.softmax(attn_score, 1)
    attn_prob = self.attention_probs_dropout(attn_prob)

    # attention output
    attn_vec = torch.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)


# In[ ]:


class MultiHeadAttn(torch.nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, 
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output

class RelMultiHeadAttn(torch.nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None])                     .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


# In[ ]:


def CBR(x, out_layer, kernel, stride, dilation):
    x = torch.nn.Conv1d(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)
    x = torch.nn.BatchNormalization()(x)
    x = torch.nn.functional.Activation("relu")(x)
    return x

def se_block(x_in, layer_n):
    x = torch.nn.GlobalAveragePooling1D()(x_in)
    x = torch.nn.Dense(layer_n//8, activation="relu")(x)
    x = torch.nn.Dense(layer_n, activation="sigmoid")(x)
    x_out=torch.nn.Multiply()([x_in, x])
    return x_out

def resblock(x_in, layer_n, kernel, dilation, use_se=True):
    x = CBR(x_in, layer_n, kernel, 1, dilation)
    x = CBR(x, layer_n, kernel, 1, dilation)
    if use_se:
        x = se_block(x, layer_n)
    x = torch.nn.Add()([x_in, x])
    return x


# In[ ]:


class TimeDistributed(torch.nn.Module):
    def __init__(self, layer, time_steps, *args):        
        super(TimeDistributed, self).__init__()
        
        self.layers = nn.ModuleList([layer(*args) for i in range(time_steps)])

    def forward(self, x):

        batch_size, time_steps, C, H, W = x.size()
        output = torch.tensor([])
        for i in range(time_steps):
          output_t = self.layers[i](x[:, i, :, :, :])
          output_t  = y.unsqueeze(1)
          output = torch.cat((output, output_t ), 1)
        return output


# # Transformer Hybrid
# 
# ---
# 
# Notes pending
# 
# ---

# In[ ]:


class TransformerXLHybridEncoder(torch.nn.Module):
  def __init__(self,
               n_token,
               n_layer,
               d_model,
               n_head,
               d_head,
               d_inner,
               dropout,
               dropout_att,
               attn_type,
               bi_data,
               is_training,
               initializer,
               mem_len=None,
               same_length=False,
               clamp_len=-1,
               untie_r=False,
               use_tpu=True,
               reuse_len=None,
               ff_activation='relu',
               use_cls_mask=False,
               **kwargs):

    super(TransformerXLHybridEncoder, self).__init__(**kwargs)

    self.n_token = n_token
    self.initializer = initializer
    self.attn_type = attn_type
    self.n_layer = n_layer
    self.d_model = d_model
    self.n_head = n_head
    self.d_head = d_head
    self.d_inner = d_inner
    self.ff_activation = ff_activation
    self.untie_r = untie_r
    self.use_tpu = use_tpu
    self.dropout = dropout
    self.dropout_att = dropout_att

    self.mem_len = mem_len
    self.reuse_len = reuse_len
    self.bi_data = bi_data
    self.clamp_len = clamp_len
    self.same_length = same_length
    self.use_cls_mask = use_cls_mask

  def build(self, unused_input_shapes):
    
    torch_float = torch.float32
    tf_float = torch_float
    self.inp = torch.nn.Input((64, 1))
    self.cbr = CBR(self.inp, 64, 7, 1, 1)
    self.embedding_lookup = EmbeddingLookup(
        n_token=self.n_token,
        d_embed=self.d_model,
        initializer=self.initializer,
        dtype=self.tf_float,
        name='embedding1')

    self.h_dropout = torch.nn.Dropout(p=self.dropout)
    self.g_dropout = torch.nn.Dropout(p=self.dropout)

    if self.untie_r:
      self.r_w_bias = (
          self.add_weight(
              'r_w_bias',
              shape=[self.n_layer, self.n_head, self.d_head],
              dtype=self.torch_float,
              initializer=self.initializer))
      self.r_r_bias = (
          self.add_weight(
              'r_r_bias',
              shape=[self.n_layer, self.n_head, self.d_head],
              dtype=self.torch_float,
              initializer=self.initializer))
      self.r_s_bias = (
          self.add_weight(
              'r_s_bias',
              shape=[self.n_layer, self.n_head, self.d_head],
              dtype=self.torch_float,
              initializer=self.initializer))
    else:
      self.r_w_bias = (
          self.add_weight(
              'r_w_bias',
              shape=[self.n_head, self.d_head],
              dtype=self.torch_float,
              initializer=self.initializer))
      self.r_r_bias = (
          self.add_weight(
              'r_r_bias',
              shape=[self.n_head, self.d_head],
              dtype=self.torch_float,
              initializer=self.initializer))
      self.r_s_bias = (
          self.add_weight(
              'r_s_bias', [self.n_head, self.d_head],
              dtype=self.torch_float,
              initializer=self.initializer))

    self.seg_embed = self.add_weight(
        'seg_embed', [self.n_layer, 2, self.n_head, self.d_head],
        dtype=self.torch_float,
        initializer=self.initializer)

    self.mask_emb = self.add_weight(
        'mask_emb/mask_emb', shape=[1, 1, self.d_model], dtype=self.torch_float)

    self.emb_dropout = torch.nn.Dropout(p=self.dropout)
    self.fwd_position_embedding = PositionalEmbedding(self.d_model)
    self.fwd_td                 = TimeDistributed(self.fwd_position_embedding, time_steps=20)
    self.fwd_lstm               = torch.nn.LSTM(128)(self.fwd_td)
    self.hidden_vect_1 = (
        Variable(torch.zeros(1, 1, hidden_size)),
        Variable(torch.zeros(1, 1, hidden_size)))
    self.output1, self.hidden1 = self.fwd_lstm(Variable(torch.rand(1, 5, 10)), hidden_vect_1)
    self.bwd_position_embedding = PositionalEmbedding(self.d_model)
    self.bwd_td                 = TimeDistributed(self.bwd_position_embedding, time_steps=20)
    self.bwd_lstm               = torch.nn.LSTM(128)(self.bwd_td)
    self.hidden_vect_1 = (
        Variable(torch.zeros(1, 1, hidden_size)),
        Variable(torch.zeros(1, 1, hidden_size)))
    self.output2, self.hidden2 = self.bwd_lstm(Variable(torch.rand(1, 5, 10)), hidden_vect_1)

    self.rel_multihead_layers = []
    self.h_positionwise_ffn_layers = []
    self.layer_norm_layers = []
    for i in range(self.n_layer):
      self.rel_multihead_layers.append(
          RelMultiHeadAttn(
              d_model=self.d_model,
              dropout=self.dropout,
              n_head=self.n_head,
              d_head=self.d_head,
              name='layer_%d/rel_attn' % (i)))
      self.h_positionwise_ffn_layers.append(
          PositionwiseFF(
              d_model=self.d_model,
              d_inner=self.d_inner,
              dropout=self.dropout,
              kernel_initializer=self.initializer,
              activation_type=self.ff_activation,
              name='layer_%d/ff' % (i)))

    self.output_dropout = torch.nn.Dropout(p=self.dropout)
    
    def __call__(self,
               inp_k,
               seg_id=None,
               input_mask=None,
               mems=None,
               perm_mask=None,
               target_mapping=None,
               inp_q=None,
               **kwargs):
    # Uses dict to feed inputs into call() in order to keep mems as a python
    # list.
        inputs = {
        'inp_k': inp_k,
        'seg_id': seg_id,
        'input_mask': input_mask,
        'mems': mems,
        'perm_mask': perm_mask,
        'target_mapping': target_mapping,
        'inp_q': inp_q
    }
    return super(TransformerXLModel, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    """Implements call() for the layer."""
    inp_k = inputs['inp_k']
    seg_id = inputs['seg_id']
    input_mask = inputs['input_mask']
    mems = inputs['mems']
    perm_mask = inputs['perm_mask']
    target_mapping = inputs['target_mapping']
    inp_q = inputs['inp_q']

    new_mems = []

    bsz = torch.shape(inp_k)[1]

    qlen = inp_k.shape.as_list()[0]

    mlen = mems[0].shape.as_list()[0] if mems is not None else 0
    klen = mlen + qlen

    ##### Attention mask
    # causal attention mask
    if self.attn_type == 'uni':
      attn_mask = _create_mask(qlen, mlen, self.tf_float, self.same_length)
      # pylint: enable=protected-access
      attn_mask = attn_mask[:, :, None, None]
    elif self.attn_type == 'bi':
      attn_mask = None
    else:
      raise ValueError('Unsupported attention type: {}'.format(self.attn_type))

    # data mask: input mask & perm mask
    if input_mask is not None and perm_mask is not None:
      data_mask = input_mask[None] + perm_mask

    elif input_mask is not None and perm_mask is None:
      data_mask = input_mask[None]
    elif input_mask is None and perm_mask is not None:
      data_mask = perm_mask
    else:
      data_mask = None

    if data_mask is not None:
      # all mems can be attended to
      mems_mask = torch.zeros([tf.shape(data_mask)[0], mlen, bsz],
                           dtype=self.tf_float)
      data_mask = torch.cat([mems_mask, data_mask], 1)
      if attn_mask is None:
        attn_mask = data_mask[:, :, :, None]
      else:
        attn_mask += data_mask[:, :, :, None]

    if attn_mask is not None:
      attn_mask = torch.cast(attn_mask > 0, dtype=self.tf_float)

    if attn_mask is not None:
      non_tgt_mask = -torch.eye(qlen, dtype=self.tf_float)
      non_tgt_mask = torch.cat(
          [tf.zeros([qlen, mlen], dtype=self.tf_float), non_tgt_mask], axis=-1)
      non_tgt_mask = torch.cast(
          (attn_mask + non_tgt_mask[:, :, None, None]) > 0, dtype=self.tf_float)
    else:
      non_tgt_mask = None

    word_emb_k = self.embedding_lookup(inp_k)

    if inp_q is not None:
      if target_mapping is not None:
        word_emb_q = torch.tile(self.mask_emb,
                             [tf.shape(target_mapping)[0], bsz, 1])
      else:
        inp_q_ext = inp_q[:, :, None]
        word_emb_q = inp_q_ext * self.mask_emb + (1 - inp_q_ext) * word_emb_k

    output_h = self.h_dropout(word_emb_k)
    output_g = None
    if inp_q is not None:
      output_g = self.g_dropout(word_emb_q)

    ##### Segment embedding
    if seg_id is not None:

      # Convert `seg_id` to one-hot `seg_mat`

      mem_pad = torch.zeros([mlen, bsz], dtype=tf.int32)

      cat_id = torch.concat([mem_pad, seg_id], 0)

      if self.use_cls_mask:
        # `1` indicates not in the same segment [qlen x klen x bsz]
        # seg_id: [qlen x bsz] & cat_id: [klen x bsz]
        cls_mat = torch.logical_or(
            torch.equal(seg_id, tf.constant([data_utils.SEG_ID_CLS]))[:, None],
            torch.equal(cat_id, tf.constant([data_utils.SEG_ID_CLS]))[None, :])
        seg_mat = torch.equal(seg_id[:, None], cat_id[None, :])
        seg_mat = torch.logical_or(cls_mat, seg_mat)
      else:
        seg_mat = tf.logical_not(tf.equal(seg_id[:, None], cat_id[None, :]))
    else:
      seg_mat = None

    dtype = self.tf_float
    freq_seq = tf.range(0, self.d_model, 2.0)
    if dtype is not None and dtype != tf.float32:
      freq_seq = tf.cast(freq_seq, dtype=self.dtype)

    if self.attn_type == 'bi':
      beg, end = klen, -qlen
    elif self.attn_type == 'uni':
      beg, end = klen, -1
    else:
      raise ValueError('Unknown `attn_type` {}.'.format(self.attn_type))

    if self.bi_data:
      fwd_pos_seq = torch.range(beg, end, -1.0)
      bwd_pos_seq = torch.range(-beg, -end, 1.0)

      if dtype is not None and dtype != tf.float32:
        fwd_pos_seq = torch.cast(fwd_pos_seq, dtype=dtype)
        bwd_pos_seq = torxh.cast(bwd_pos_seq, dtype=dtype)

      if self.clamp_len > 0:
        fwd_pos_seq = torch.clip_by_value(fwd_pos_seq, -self.clamp_len,
                                       self.clamp_len)
        bwd_pos_seq = torch.clip_by_value(bwd_pos_seq, -self.clamp_len,
                                       self.clamp_len)

      if bsz is not None:
        fwd_pos_emb = self.fwd_position_embedding(fwd_pos_seq, bsz // 2)
        bwd_pos_emb = self.bwd_position_embedding(bwd_pos_seq, bsz // 2)
      else:
        fwd_pos_emb = self.fwd_position_embedding(fwd_pos_seq, None)
        bwd_pos_emb = self.bwd_position_embedding(bwd_pos_seq, None)

      pos_emb = tf.concat([fwd_pos_emb, bwd_pos_emb], axis=1)
    else:
      fwd_pos_seq = tf.range(beg, end, -1.0)
      if dtype is not None and dtype != tf.float32:
        fwd_pos_seq = tf.cast(fwd_pos_seq, dtype=dtype)
      if self.clamp_len > 0:
        fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -self.clamp_len,
                                       self.lamp_len)

      pos_emb = self.fwd_position_embedding(fwd_pos_seq, bsz)

    pos_emb = self.emb_dropout(pos_emb)

    if mems is None:
      mems = [None] * self.n_layer
    for i in range(self.n_layer):
      # cache new mems
      new_mems.append(
          _cache_mem(output_h, mems[i], self.mem_len, self.reuse_len))
      # pylint: enable=protected-access

      # segment bias
      if seg_id is None:
        r_s_bias_i = None
        seg_embed_i = None
      else:
        r_s_bias_i = self.r_s_bias if not self.untie_r else self.r_s_bias[i]
        seg_embed_i = self.seg_embed[i]

      ffn_layer = self.h_positionwise_ffn_layers[i]
      attention_layer = self.rel_multihead_layers[i]
      output_h, output_g = attention_layer(
          h=output_h,
          g=output_g,
          r=pos_emb,
          r_w_bias=self.r_w_bias if not self.untie_r else self.r_w_bias[i],
          r_r_bias=self.r_r_bias if not self.untie_r else self.r_r_bias[i],
          seg_mat=seg_mat,
          r_s_bias=r_s_bias_i,
          seg_embed=seg_embed_i,
          attn_mask_h=non_tgt_mask,
          attn_mask_g=attn_mask,
          mems=mems[i],
          target_mapping=target_mapping)
      output_h = ffn_layer(output_h)
      if output_g is not None:
        output_g = ffn_layer(output_g)

    if inp_q is not None:
      output = output_g
    else:
      output = output_h

    return output, new_mems, None


# In[ ]:


class AdaptiveEmbedding(torch.nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                 sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = [cutoffs] + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + [self.cutoffs]

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            for i in range(1,self.n_token):
                self.emb_layers.append(
                    nn.Embedding(n_token, d_embed, sparse=(sample_softmax > 0))
                )
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx-l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            for i in range(1,self.n_token):
                embed = []
                embed.append(self.emb_layers[i](inp))
                if self.d_proj != self.d_embed:
                    embed = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj],
                                   dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed


# In[ ]:


class GPT2OptimizedDecoderLayer(torch.nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, hidden_size, **kwargs):
        super(GPT2OptimizedDecoderLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseLSTMFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))
        self.dec_attn2 = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_gru_ff = PositionwiseGRUFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))
        
    
        ###### ATTENTION #####################################
        # self.dec_attn3 = GPT2ParallelSelfAttn(hidden_size, )
        ###MUST WORK ON GPT2 ATTENTION########################

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_gru_ff(output)
        output = self.dec_attn2(dec_inp, attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class RelLearnableDecoderLayer(torch.nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head,
                                                  dropout, **kwargs)
        self.pos_gru_ff = PositionwiseGRUFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))
        self.dec_attn2 = RelLearnableMultiHeadAttn(n_head, d_model, d_head,
                                                  dropout, **kwargs)
        self.pos_ff = PositionwiseLSTMFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_gru_ff(output)
        output = self.dec_attn2(dec_inp, r_emb, r_w_bias, r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class RelPartialLearnableDecoderLayer(torch.nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                                                         d_head, dropout,
                                                         **kwargs)
        self.pos_ff = PositionwiseLSTMFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


# In[ ]:


class Transformer(torch.nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, dtype, attention_dropout_prob, output_dropout_prob, 
                 init_method, bi_data, tie_weight=True, d_embed=None,
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=10, ext_len=10, mem_len=10,
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=2, clamp_len=-1,
                 sample_softmax=-1):
        super(Transformer, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.drop = nn.Dropout(dropout)

        self.tie_weight = tie_weight
        self.tie_projs = tie_projs
        self.div_val = div_val

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type
        
        self.word_emb = PositionalEmbedding(d_model)

        self.layers = nn.ModuleList()
        self.layers.append(
            TransformerXLHybridEncoder(
               n_token,
               n_layer,
               d_model,
               n_head,
               d_head,
               d_inner,
               dropout,
               dropatt,
               bi_data,
               attn_type,
               is_training=True,
               initializer=torch.optim.SGD,
            )
        )
        self.layers.append(
            TransformerXLHybridEncoder(
               n_token,
               n_layer,
               d_model,
               n_head,
               d_head,
               d_inner,
               dropout,
               dropatt,
               bi_data,               
               attn_type,
               is_training=True,
               initializer=torch.optim.SGD,
            )
        )
        self.layers.append(
            TransformerXLHybridEncoder(
               n_token,
               n_layer,
               d_model,
               n_head,
               d_head,
               d_inner,
               dropout,
               dropatt,
               bi_data,
               attn_type,
               is_training=True,
               initializer=torch.optim.SGD,
            )
        )
        # the default attention
        if attn_type == 0:
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        # learnable embeddings
        elif attn_type == 1:
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        # absolute embeddings
        elif attn_type in [2, 3]:
            for i in range(n_layer):
                self.layers.append(
                    GPT2OptimizedDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm, hidden_size=16)
                )
        
        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)
        
        
        
        # use adaptive softmax (including standard softmax)
        else:
            if tie_weight:
                emb_layers = [i.weight for i in AdaptiveEmbedding(d_model, d_head, d_inner, n_head).emb_layers]
            else:
                emb_layers = None

            emb_projs = AdaptiveEmbedding(d_model, d_head, d_inner, n_head).emb_projs

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1
    cutoffs=[]
    
    def _create_params(self):
        # default attention
        cutoffs=[]
        if self.attn_type == 0:
            self.pos_emb = AdaptiveEmbedding(self.n_token, self.d_embed, self.d_model, cutoffs,
                                          div_val=self.div_val)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        # learnable
        elif self.attn_type == 1:
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head))
        # absolute standard
        elif self.attn_type == 2:
            self.pos_emb = PositionalEmbedding(self.d_model)
        # absolute deeper SA
        elif self.attn_type == 3:
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.d_model))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inp, mems=None):
        qlen, bsz = dec_inp.size()
        true_size = 7

        word_emb = PositionalEmbedding(dec_inp)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen

        # absolute
        if self.attn_type == 2:
            pos_seq = torch.LongTensor(torch.arange(klen - 1, -1, -1.0, dtype=torch.long))
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq, 64)

            core_out = self.drop(pos_emb[-qlen:])
            hids = []
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and len(mems_i) and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = core_out
                hids.append(core_out)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and len(mems_i) and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(mlen-cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, qlen, mlen)

        return core_out, new_mems

    def forward(self, data, target, mems):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if mems is None:
            mems = self.init_mems()

        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, mems=None)

        pred_hid = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(self.word_emb, self.out_layer.bias, target,
                                  pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
            loss = loss.view(tgt_len, -1)


# ## Data wrangling 

# In[ ]:


train = pd.read_csv('../input/preprocessing-for-m5/training.csv')
test = pd.read_csv('../input/preprocessing-for-m5/test.csv')


# In[ ]:


from sklearn.model_selection import train_test_split as T
from sklearn.preprocessing import OneHotEncoder as OHE
X_train, y_train = T(train,test_size=0.1)
X_test, y_test = T(test,test_size=0.1)


# In[ ]:


X = Transformer(
    n_token=500,
    n_layer=18,
    n_head=8,
    d_model=10,
    d_head=8,
    d_inner=12,
    dropout=0.2,
    dropatt=0.2,
    dtype=torch.float32,
    attention_dropout_prob=0.15,
    output_dropout_prob=0.175,
    init_method=torch.optim.SGD,
    bi_data=10
)


# In[ ]:


X


# In[ ]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(X.parameters(), lr=0.001, momentum=0.9)
train = train.drop(['id'], axis=1)
train = train.drop(['item_id'], axis=1)
train = train.drop(['dept_id'], axis=1)
train = train.drop(['cat_id'], axis=1)
train = train.drop(['store_id'], axis=1)
train = train.drop(['state_id'], axis=1)


# In[ ]:


X_train = X_train.drop(['id'], axis=1)
X_train = X_train.drop(['item_id'], axis=1)
X_train = X_train.drop(['dept_id'], axis=1)
X_train = X_train.drop(['cat_id'], axis=1)
X_train = X_train.drop(['store_id'], axis=1)
X_train = X_train.drop(['state_id'], axis=1)
y_train = y_train.drop(['id'], axis=1)
y_train = y_train.drop(['item_id'], axis=1)
y_train = y_train.drop(['dept_id'], axis=1)
y_train = y_train.drop(['cat_id'], axis=1)
y_train = y_train.drop(['store_id'], axis=1)
y_train = y_train.drop(['state_id'], axis=1)
X_train.to_csv('TRAIN_X.csv', index=False)
del X_train
y_train.to_csv('TRAIN_Y.csv', index=False)
del y_train


# # Alternate (simpler) structure
# 
# Basically take out all the hijinks from the model.

# In[ ]:


class Transformer(torch.nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, dtype, attention_dropout_prob, output_dropout_prob, 
                 init_method, bi_data, tie_weight=True, d_embed=None,
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=10, ext_len=10, mem_len=10,
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=2, clamp_len=-1,
                 sample_softmax=-1):
        super(Transformer, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.drop = nn.Dropout(dropout)

        self.tie_weight = tie_weight
        self.tie_projs = tie_projs
        self.div_val = div_val

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type
        
        self.word_emb = PositionalEmbedding(d_model)

        self.layers = nn.ModuleList()
        self.layers.append(
            TransformerXLHybridEncoder(
               n_token,
               n_layer,
               d_model,
               n_head,
               d_head,
               d_inner,
               dropout,
               dropatt,
               bi_data,
               attn_type,
               is_training=True,
               initializer=torch.optim.SGD,
            )
        )
        # the default attention
        if attn_type == 0:
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        # learnable embeddings
        elif attn_type == 1:
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        # absolute embeddings
        elif attn_type in [2, 3]:
            for i in range(n_layer):
                self.layers.append(
                    GPT2OptimizedDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm, hidden_size=16)
                )
        
        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)
        
        
        
        # use adaptive softmax (including standard softmax)
        else:
            if tie_weight:
                emb_layers = [i.weight for i in AdaptiveEmbedding(d_model, d_head, d_inner, n_head).emb_layers]
            else:
                emb_layers = None

            emb_projs = AdaptiveEmbedding(d_model, d_head, d_inner, n_head).emb_projs

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1
    cutoffs=[]
    
    def _create_params(self):
        # default attention
        cutoffs=[]
        if self.attn_type == 0:
            self.pos_emb = AdaptiveEmbedding(self.n_token, self.d_embed, self.d_model, cutoffs,
                                          div_val=self.div_val)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        # learnable BUT slower
        """
        elif self.attn_type == 1:
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head))"""
        # absolute standard and faster
        if self.attn_type == 2 or self.attn_type == 1:
            self.pos_emb = PositionalEmbedding(self.d_model)
        # absolute deeper SA
        elif self.attn_type == 3:
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.d_model))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inp, mems=None):
        qlen, bsz = dec_inp.size()
        true_size = 7

        word_emb = PositionalEmbedding(dec_inp)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen

        # absolute
        if self.attn_type == 2:
            pos_seq = torch.LongTensor(torch.arange(klen - 1, -1, -1.0, dtype=torch.long))
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq, 64)

            core_out = self.drop(pos_emb[-qlen:])
            hids = []
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and len(mems_i) and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = core_out
                hids.append(core_out)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and len(mems_i) and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(mlen-cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, qlen, mlen)

        return core_out, new_mems

    def forward(self, data, target, mems):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if mems is None:
            mems = self.init_mems()

        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, mems=None)

        pred_hid = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(self.word_emb, self.out_layer.bias, target,
                                  pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
            loss = loss.view(tgt_len, -1)


# ---
# 
# # My other works for this competition
# 
# Please check out these other works by me for this competition:
# + https://www.kaggle.com/nxrprime/thoughts-on-creating-a-good-nn-model
# + https://www.kaggle.com/nxrprime/preprocessing-fe
# + https://www.kaggle.com/nxrprime/loss-functions
# + https://www.kaggle.com/nxrprime/m5-feather-files-for-fast-loading-48x-faster
