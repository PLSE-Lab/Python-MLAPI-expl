#!/usr/bin/env python
# coding: utf-8

# ## Credit to Konrad Banachewicz.
# ### BTW, the magic is risky for the private lb.

# In[ ]:


import pandas as pd
x = pd.read_csv('../input/beat-the-benchmark-snaive/btb_snaive.csv')
for i in range(1, 29):
    x[f'F{i}'] *= 1.04
x.to_csv('sub.csv',index=False)

