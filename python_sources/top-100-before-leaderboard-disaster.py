#!/usr/bin/env python
# coding: utf-8

# The labels for the public leaderboard have now been released. <br/>
# Leaderboard is under attack. <br/>
# I tried to keep the list of top 100 before it. Please refer. <br/>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# In[ ]:


df_top = pd.read_csv('/kaggle/input/m5-top100/m5_top100_utf8.csv')
pd.set_option('display.max_rows', df_top.shape[0]+1)
df_top.style.hide_index()


# In[ ]:




