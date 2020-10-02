#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


seq = pd.read_csv("/kaggle/input/genetic-sequences-for-the-sarscov2-coronavirus/Genetic-Sequences-for-the-SARS-CoV-2-Coronavirus.tsv",
                 sep = '\t', engine='python', quoting=3)

seq.head().T


# In[ ]:


sns.distplot(seq['"Length"'])


# In[ ]:


sns.lineplot(data=seq, x='"CollectionDate"', y = '"Length"')


# In[ ]:


seq.describe(include='all')


# In[ ]:


seq['"Sequence"']


# In[ ]:


type(seq['"Sequence"'][0])


# In[ ]:


list(seq['"Sequence"'][0])


# In[ ]:




