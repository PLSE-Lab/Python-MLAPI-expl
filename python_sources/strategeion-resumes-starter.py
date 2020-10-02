#!/usr/bin/env python
# coding: utf-8

# # Strategeion Resumes Starter

# In[ ]:


from shutil import copyfile
copyfile(src = "../input/fairness.py", dst = "../working/fairness.py")
import fairness
import pandas as pd


# In[ ]:


data = pd.read_csv("../input/resumes_development.csv", index_col=0)
data.head()

