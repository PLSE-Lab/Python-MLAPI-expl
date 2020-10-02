#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from IPython.display import display, Markdown


# In[ ]:


train = pd.read_csv('/kaggle/input/birdsong-recognition/train.csv')
print ('There are {} species'.format(train.species.nunique()))


# In[ ]:


for i, s in enumerate(train.species.unique()):
    ss = s.replace("'", "")
    ss = ss.replace(" ", "_")
    display(Markdown(f'[{str(i).zfill(2)}: {s}](https://www.allaboutbirds.org/guide/{ss}/overview)'))


# In[ ]:




