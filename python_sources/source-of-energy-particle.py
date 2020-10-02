#!/usr/bin/env python
# coding: utf-8

# Hello Testing

# In[28]:


from trackml.dataset import load_event
from IPython.display import display
import tensorflow as tf
import pandas as pd
import numpy as np
import glob

train = np.unique([p.split('-')[0] for p in sorted(glob.glob('../input/train_1/**'))])
test = np.unique([p.split('-')[0] for p in sorted(glob.glob('../input/test/**'))])
det = pd.read_csv('../input/detectors.csv')
sub = pd.read_csv('../input/sample_submission.csv')
print(len(train), len(test), len(det), len(sub))


# In[27]:


for e in train:
    hits, cells, particles, truth = load_event(e)
    print(len(hits), len(cells), len(particles), len(truth))
    display(hits.head(2))
    display(cells.head(2))
    display(particles.head(2))
    display(truth.head(2))
    break


# In[ ]:




