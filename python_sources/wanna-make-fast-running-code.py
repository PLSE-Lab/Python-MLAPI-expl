#!/usr/bin/env python
# coding: utf-8

# [I read this kernel.](https://www.kaggle.com/harupy/m5-baseline)
# 
# My code isn't as good as normal code. But I'm trying to make faster code because even the Normal code spend a lot of time:)
# 
# ### __We require very fast code because it handles very large data.__
# 
# ### If you have any idea after looking at my code, please tell me on comment.
# 
# I share my thought.(But it's so not good.)

# In[ ]:


from typing import Any, Dict
import numpy as np
import itertools as it
import gc
from tqdm import tqdm
from numba import jit
import pandas as pd


# In[ ]:


def read_data(n_rows: int) -> pd.DataFrame:
    data = pd.read_pickle("../input/walmartbasedata/data2.pickle").loc[:n_rows]
    data = data[data["part"] != "evaluation"]
    return data
data = read_data(1_000)
DAYS_PRED = 28


# In[ ]:


data


# ## Normal code(15.9s)
# 
# The two codes do the same thing.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'df = data.copy()\nmemo = df.groupby(["id"])["demand"]\nfor diff in [0, 1, 2]:\n    shift = DAYS_PRED + diff\n    df[f"shift_t{shift}"] = memo.transform(\n        lambda x: x.shift(shift)\n    )\n\nfor size in [7, 30, 60, 90, 180]:\n    df[f"rolling_std_t{size}"] = memo.transform(\n        lambda x: x.shift(DAYS_PRED).rolling(size).std()\n    )\n\nfor size in [7, 30, 60, 90, 180]:\n    df[f"rolling_mean_t{size}"] = memo.transform(\n        lambda x: x.shift(DAYS_PRED).rolling(size).mean()\n    )\n\ndf["rolling_skew_t30"] = memo.transform(\n    lambda x: x.shift(DAYS_PRED).rolling(30).skew()\n)\ndf["rolling_kurt_t30"] = memo.transform(\n    lambda x: x.shift(DAYS_PRED).rolling(30).kurt()\n)\ndel memo\ndf')


# ## My code(25.9s)

# In[ ]:


get_ipython().run_cell_magic('time', '', 'master_id = []\nmemo = data.groupby("id")["demand"]\n# id_list = sorted(data["id"].unique().tolist())\nflag = False\nDAYS_PRED = 28\nfor id in tqdm(memo):\n    x = id[1]\n    id = pd.DataFrame(id[1])\n    for diff in [0, 1, 2]:\n        shift = DAYS_PRED + diff\n        id[f"shift_t{shift}"] = x.shift(shift)\n    for size in [7, 30, 60, 90, 180]:\n        id[f"rolling_std_t{size}"] = id["shift_t28"].rolling(size).std()\n        id[f"rolling_mean_t{size}"] = id["shift_t28"].rolling(size).mean()\n    id["rolling_skew_t30"] = id["shift_t28"].rolling(30).skew()\n    id["rolling_kurt_t30"] = id["shift_t28"].rolling(30).kurt()\n    master_id.append(id)\npd.concat(master_id)')


# ## Please share how to make better code.
# 
# Thank you for reading my kernel.(I'm not at English.)
