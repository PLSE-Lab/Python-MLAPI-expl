#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


get_ipython().system('pip install --upgrade bamboolib>=1.2.0')


# In[ ]:


import bamboolib as bam
bam.enable()


# In[ ]:


netflix = pd.read_csv("/kaggle/input/netflix-shows/netflix_titles_nov_2019.csv")


# In[ ]:


bam.show(netflix)


# In[ ]:




