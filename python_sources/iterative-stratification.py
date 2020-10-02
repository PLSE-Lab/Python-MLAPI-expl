#!/usr/bin/env python
# coding: utf-8

# ### In this competition, we have three targets for each sample, thus I think using iterative stratifications (https://github.com/trent-b/iterative-stratification) is helpful according to previous competition:
# imet top 1 solution https://www.kaggle.com/c/imet-2019-fgvc6/discussion/94687

# In[ ]:


import numpy as np
import pandas as pd
get_ipython().system('pip install iterative-stratification')


# In[ ]:


#get data
nfold = 5
seed = 12

train_df = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')
train_df['id'] = train_df['image_id'].apply(lambda x: int(x.split('_')[1]))

X, y = train_df[['id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values[:,0], train_df.values[:,1:]

train_df['fold'] = np.nan

#split data
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
mskf = MultilabelStratifiedKFold(n_splits=nfold, random_state=seed)
for i, (_, test_index) in enumerate(mskf.split(X, y)):
    train_df.iloc[test_index, -1] = i
    
train_df['fold'] = train_df['fold'].astype('int')


# In[ ]:


#output
train_df.to_csv('train_with_fold.csv', index = False)

