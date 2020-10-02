#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path.ls = lambda self: list(self.glob('*'))


# In[ ]:


path = Path('../input/covid-chest-xray')


# ## Metadata

# In[ ]:


df = pd.read_csv(path/'metadata.csv')


# In[ ]:


df


# In[ ]:


survivors = df.groupby('Patientid').agg({'survival': [lambda series: series.at[0]]})
survivors = survivors.reset_index()


# In[ ]:


survivors


# ## X-ray

# In[ ]:


images = path/'images'


# In[ ]:


def ceildiv(a, b):
    return -(-a // b)

def plots_from_files(imspaths, figsize=(10,5), rows=1, titles=None, maintitle=None):
    """Plot the images in a grid"""
    f = plt.figure(figsize=figsize)
    if maintitle is not None: plt.suptitle(maintitle, fontsize=10)
    for i in range(len(imspaths)):
        sp = f.add_subplot(rows, ceildiv(len(imspaths), rows), i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        img = plt.imread(imspaths[i])
        plt.imshow(img)


# In[ ]:


imspaths = np.random.choice(images.ls(), size=9)
plots_from_files(imspaths, rows=3, figsize=(20,20))


# to be continued.

# In[ ]:




