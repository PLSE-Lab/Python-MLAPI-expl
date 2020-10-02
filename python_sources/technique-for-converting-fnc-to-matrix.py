#!/usr/bin/env python
# coding: utf-8

# # Convert FNC data to Connectivity Matrices Efficiently
# 
# The explanation of this notebook is here: https://www.kaggle.com/c/trends-assessment-prediction/discussion/147128
# 
# ![Converting](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F473234%2Ffd306599c498e101b6a7558719e3611d%2Fconvert_fnc_to_matrix.png?generation=1588174928673267&alt=media)

# ### preparation

# In[ ]:


import os
import sys
import shutil
import time
import random
import re
import gc

from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

from functools import partial
from itertools import chain
from collections import Counter

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


COMPETITION_NAME = "trends-assessment-prediction"
ROOT = Path(".").resolve().parents[0]

INPUT_ROOT = ROOT / "input"
RAW_DATA = INPUT_ROOT / COMPETITION_NAME
TRAIN_IMAGES = RAW_DATA / "fMRI_train"
TEST_IMAGES = RAW_DATA / "fMRI_test"


# In[ ]:


fnc = pd.read_csv(RAW_DATA / "fnc.csv")
icn_numbers = pd.read_csv(RAW_DATA / "ICN_numbers.csv")
loading = pd.read_csv(RAW_DATA / "loading.csv")
reveal_ID_site2 = pd.read_csv(RAW_DATA / "reveal_ID_site2.csv")

train_scores = pd.read_csv(RAW_DATA / "train_scores.csv")
sample_sub = pd.read_csv(RAW_DATA / "sample_submission.csv")


# In[ ]:


def convert_fnc_feature_to_symmetric_matrix(
    fnc_arr: np.ndarray, n_fnc: int, diag_comp: Union[int, float, str]=0. 
) -> np.ndarray:
    """
    Convet fnc features to Matrix.
    
    This matrix is symmetric matrix. The bellow is a toy sample.
    Example:
      input: shape = (2, 4)
        [
          [a_{f2_vs_f1}, a_{f3_vs_f1}, a_{f4_vs_f1}. a_{f3_vs_f2}, a_{f4_vs_f2}, a_{f4_vs_f3}],
          [b_{f2_vs_f1}, b_{f3_vs_f1}, b_{f4_vs_f1}. b_{f3_vs_f2}, b_{f4_vs_f2}, b_{f4_vs_f3}],
        ]
      -> output: shape = (2, 4, 4)
        [
          [
            [      0     , a_{f2_vs_f1}, a_{f3_vs_f1}, a_{f4_vs_f1}],
            [a_{f2_vs_f1},        0    , a_{f3_vs_f2}, a_{f4_vs_f2}],
            [a_{f3_vs_f1}, a_{f3_vs_f2},        0    , a_{f4_vs_f3}],
            [a_{f4_vs_f1}, a_{f4_vs_f2}, a_{f4_vs_f3},        0    ],
          ],
          [
            [      0     , a_{f2_vs_f1}, a_{f3_vs_f1}, a_{f4_vs_f1}],
            [a_{f2_vs_f1},        0    , a_{f3_vs_f2}, a_{f4_vs_f2}],
            [a_{f3_vs_f1}, a_{f3_vs_f2},        0    , a_{f4_vs_f3}],
            [a_{f4_vs_f1}, a_{f4_vs_f2}, a_{f4_vs_f3},        0    ],
          ],
        ]
    """
    n_example = fnc_arr.shape[0]
    tmp = np.repeat(np.arange(n_fnc - 1, -1, -1)[:, None], n_fnc, axis=1)
    tmp[0] = np.arange(n_fnc)
    tmp = np.cumsum(tmp, axis=0)
    idx_arr = np.triu(tmp, 1) + np.tril(tmp.T, -1)
    
    fnc_arr = np.concatenate([np.full((n_example, 1), diag_comp), fnc_arr], axis=1)
    return fnc_arr[(
        np.tile(np.arange(n_example)[:, None, None], (n_fnc, n_fnc)),
        np.repeat(idx_arr[None, ...], n_example, axis=0),
    )]


# In[ ]:


icn_numbers["FNC_name"] = ["SCN(69)"] + list(map(lambda x: x.split("_")[0], fnc.columns[1:53]))
icn_numbers


# ### cheack the order of fnc feature's column

# In[ ]:


l = []
tmp_idx = 0
n_comb = 52
fnc_columns = fnc.columns[1:].tolist()
for i, fnc_name in enumerate(icn_numbers.FNC_name[:-1]):
    l.append(["-"] * (i + 1) + fnc_columns[tmp_idx: tmp_idx + n_comb])
    tmp_idx = tmp_idx + n_comb
    n_comb -= 1
l.append(["-"] * 53)

pd.options.display.max_columns=53
pd.DataFrame(
    l,
    columns=icn_numbers.FNC_name.tolist(),
    index=icn_numbers.FNC_name.tolist(),)


# ### Converting
# 
# convert `fnc_feat_vecs`: $ \ \left(11754, \ 1378 \right) = \left(11754, \ {}_{53} C _{2} \right) \ $ to `fnc_conn_mats` : $ \ \left(11754, \ 53, \ 53 \right)$

# In[ ]:


# # remove id
fnc_feat_vecs = fnc.iloc[:, 1:].values
fnc_feat_vecs.shape


# In[ ]:


get_ipython().run_line_magic('time', '')
fnc_conn_mats = convert_fnc_feature_to_symmetric_matrix(fnc_feat_vecs, n_fnc=53, diag_comp=0)


# In[ ]:


fnc_conn_mats.shape


# For instance, check the part of ```fnc_feat_vecs``` and ```fnc_conn_mats```

# 5 tail elements of `XXX(Y)_vs_SCN(69)`of 3 examples

# In[ ]:


fnc_feat_vecs[:3, 52 - 5:52]


# In[ ]:


fnc_conn_mats[:3, 0, -5:]


# 5 tail elements of `XXX(Y)_vs_SCN(53)` of 3 examples

# In[ ]:


fnc_feat_vecs[:3, 52 + 51 - 5:52 + 51]


# In[ ]:


fnc_conn_mats[:3, 1, -5:]


# ### Visualization
# 
# Reference: https://www.kaggle.com/srsteinkamp/trends-eda#Back-to-FNC-data
# 
# This notebook is helpful for me to understand competition data. Thanks!

# In[ ]:


col_labels = icn_numbers["FNC_name"].values
fig, axes = plt.subplots(4, 2,figsize=(22.5, 52))

for i in range(8):
    mat_example = fnc_conn_mats[i]
    row_idx, col_idx = divmod(i, 2)
    ax = axes[row_idx, col_idx]
    sns.heatmap(
        mat_example, cmap='coolwarm', square=True, ax=ax, 
        xticklabels=col_labels, yticklabels=col_labels,
        cbar=False, center=0, vmin=-1, vmax=1)

    _ = ax.set_title('Mat Example {}'.format(i), fontsize=15)


# In[ ]:




