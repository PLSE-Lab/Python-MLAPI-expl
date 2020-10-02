#!/usr/bin/env python
# coding: utf-8

# https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/109408#latest-629574

# In[ ]:


import pandas as pd
from pathlib import Path

input_path = Path("../input/rsna-intracranial-hemorrhage-detection/")


# In[ ]:


get_ipython().run_line_magic('ls', '../input/rsna-intracranial-hemorrhage-detection')


# In[ ]:


df = pd.read_csv(input_path / "stage_1_train.csv")
df.head()


# In[ ]:


df.shape


# In[ ]:


df["ID"].unique().shape


# In[ ]:


df[df["ID"].duplicated(keep=False)]


# In[ ]:




