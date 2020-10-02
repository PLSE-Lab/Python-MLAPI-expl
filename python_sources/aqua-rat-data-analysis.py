#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import gzip
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
import pickle
import collections
from pandas.io.json import json_normalize
import re
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'head -c 1000 "../input/aquarat-algebra-question-answering-with-rationale/train AQuA.json"')

