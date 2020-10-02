#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/train.csv', index_col='Id')
seqs = {ix: pd.Series(x['Sequence'].split(',')) for ix, x in train.iterrows()}


# # SequenceSize

# In[ ]:


train['SequenceSize'] = [len(seq) for seq in seqs.values()]


# # Mode

# In[ ]:


train['Mode'] = [seq.value_counts().idxmax() for seq in seqs.values()]


# # LastValue

# In[ ]:


train['LastValue'] = [seq.iloc[-1] for seq in seqs.values()]


# # IsLastValueEqMode

# In[ ]:


train['IsLastValueEqMode'] = train.apply(lambda x: x['LastValue'] == x['Mode'], axis=1)


# # NDifferentValues

# In[ ]:


train['NDifferentValues'] = [seq.value_counts().shape[0] for seq in seqs.values()]


# # SeqValuesSizeMean/Max/Min

# In[ ]:


train['SeqValuesSizeMean'] = [seq.apply(lambda x: len(x)).mean() for seq in seqs.values()]
train['SeqValuesSizeMax'] = [seq.apply(lambda x: len(x)).max() for seq in seqs.values()]
train['SeqValuesSizeMin'] = [seq.apply(lambda x: len(x)).min() for seq in seqs.values()]


# # SequenceSize and NDifferentValues scatter

# In[ ]:


ax = train[train.IsLastValueEqMode].plot(kind='scatter', x='SequenceSize', y='NDifferentValues', color='Blue', figsize=(12, 12),       label='LastValue = Mode', xlim=(0, 400), ylim=(0, 200))
train[~train.IsLastValueEqMode].plot(kind='scatter', x='SequenceSize', y='NDifferentValues', color='Red', ax=ax,       label='LastValue <> Mode', alpha=.5)


# In[ ]:




