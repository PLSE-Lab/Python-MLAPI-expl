#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv("../input/sample_predict.csv")
df.to_csv("submission.csv", index=False)

