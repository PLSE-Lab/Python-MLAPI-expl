#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

df = pd.DataFrame([[i, i + 2] for i in range(101)], columns = ['id', 'new_number'])

df


# In[ ]:


df.to_csv('submission.csv', index = False, header = True)
# f = open('/kaggle/working/submission.csv', 'r')
# print(f.read())

import os
os.chdir(r'/kaggle/working')
from IPython.display import FileLink
FileLink(r'submission.csv')

