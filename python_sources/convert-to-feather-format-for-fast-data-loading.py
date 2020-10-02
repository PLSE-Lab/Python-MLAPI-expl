#!/usr/bin/env python
# coding: utf-8

# # Liverpool ion switching: convert to feather format for fast data loading
# 
# We can load much faster with feather format, instead of using csv.
# 
# This data is uploaded to [Liverpool ion switching feather](https://www.kaggle.com/corochann/liverpool-ion-switching-feather). You can press "Add Data" button on top-right and search "liverpool-ion-switching-feather" to use this feather format data.

# In[ ]:


import os

import pandas as pd
from pathlib import Path


# It takes 3 sec to read data with the original csv format.

# In[ ]:


get_ipython().run_cell_magic('time', '', "datadir = Path('../input/liverpool-ion-switching')\ntrain = pd.read_csv(datadir / 'train.csv')\ntest = pd.read_csv(datadir / 'test.csv')\nsubmission = pd.read_csv(datadir / 'sample_submission.csv')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# --- save feather format ---\noutdir = Path('.')\nos.makedirs(str(outdir), exist_ok=True)\ntrain.to_feather(outdir / 'train.feather')\ntest.to_feather(outdir / 'test.feather')\nsubmission.to_feather(outdir / 'sample_submission.feather')")


# Let's check how fast to read the data with feather format, it takes about **only 100 milli-sec** to read data!

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# --- check reading ---\ntrain2 = pd.read_feather(outdir / 'train.feather')\ntest2 = pd.read_feather(outdir / 'test.feather')\nsubmission2 = pd.read_feather(outdir / 'sample_submission.feather')")

