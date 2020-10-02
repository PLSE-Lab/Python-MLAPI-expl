#!/usr/bin/env python
# coding: utf-8

# Here is my optimized version of "[Faster stratified cross-validation (V2)](https://www.kaggle.com/frednavruzov/faster-stratified-cross-validation-v2)".
# 
# It runs ca. 5 times faster than the original version and has such a plain-dumb logic that it is difficult to make a mistake :) At also has very low memory consumption.
# 
# I reused almost the whole original notebook so that you could test yourself.
# 
# The new code is in the function CustomStratifiedKFold2

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from copy import deepcopy


# In[ ]:


# let's define our own stratified validator "with blackjack and hookers" :)
class CustomStratifiedKFold:
    """
    Faster (yet memory-heavier) stratified cross-validation split
    Best suited for longer time-series with many different `y` groups
    """
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = random_state
        self.folds_ = [(list(), list()) for _ in range(n_splits)]
        self.randomizer_ = np.random.RandomState(random_state)
        self.groups_ = None
        self.counts_ = None
        self.s_ = None

    def split(self, X, y):
        sorted_y = pd.Series(y).reset_index(drop=True).sort_values().astype('category').cat.codes
        self.s_ = pd.Series(data=sorted_y.index.values, index=sorted_y)
        self.groups_ = self.s_.index.unique()
        self.counts_ = np.bincount(self.s_.index)

        if self.n_splits > self.counts_.min():
            raise ValueError(
                f'Cannot split {self.counts_.min()} elements in smallest group on {self.n_splits} folds'
            )

        shift = 0
        for cnt in tqdm(self.counts_, desc='processing unique strats'):
            # get array of initial data's indices
            arr = self.s_.iloc[shift:shift + cnt].values
            # shuffle data if needed
            if self.shuffle:
                self.randomizer_.shuffle(arr)
            folds = np.array_split(arr, self.n_splits)
            # extend outer folds by elements from micro-folds
            for i in range(self.n_splits):
                cp = deepcopy(folds)
                # extend val indices
                val_chunk = cp.pop(i).tolist()
                self.folds_[i][1].extend(val_chunk)
                # extend train indices
                if self.shuffle:
                    cp = self.randomizer_.permutation(cp)
                train_chunk = np.hstack(cp).tolist()
                self.folds_[i][0].extend(train_chunk)

            # shift to the next group
            shift += cnt
        assert shift == len(self.s_)

        for (t, v) in self.folds_:
            yield (
                np.array(self.randomizer_.permutation(t) if self.shuffle else t, dtype=np.int32),
                np.array(self.randomizer_.permutation(v) if self.shuffle else v, dtype=np.int32)
            )


# In[ ]:


from collections import defaultdict

class CustomStratifiedKFold2:
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits    = n_splits
        self.shuffle     = shuffle
        self.randomizer_ = np.random.RandomState(random_state)

    def split(self, X, y):
        folds = defaultdict(lambda: np.zeros(len(y), dtype=np.int8))              # global fold reference

        for i, (g, u) in enumerate(tqdm(pd.Series(y).groupby(by=y), desc='processing strats')):
            strata = u.index.values                                               # array of initial data's indices for the given strata

            l = len(strata)
            if self.n_splits > l:
                raise ValueError(f'Cannot split {l} elements in group {u} on {self.n_splits} folds')
            
            if self.shuffle:                                                      # shuffle if asked so
                self.randomizer_.shuffle(strata)
                
            tl = l // self.n_splits
            for j in range(self.n_splits):
                folds[j][strata]                =  1                              # first set the whole strata to train
                folds[j][strata[tl*j:tl*(j+1)]] =  2                              # now set the valid part 
                
        for i in range(self.n_splits):
            yield (folds[i] == 1, folds[i] == 2)


# ### Let's create some fake data to test different approaches on

# In[ ]:


N_FOLDS = 5
SEED = 42
SHUFFLE = True
NUM_UNIQUES = 1000
N = 7000000

randomizer = np.random.RandomState(SEED)
strat_column = pd.Series(randomizer.randint(0, NUM_UNIQUES, N, dtype=np.int32))
print(strat_column.nunique())
strat_column.head(10)


# In[ ]:


get_ipython().run_cell_magic('time', '', "# let's check usual StratifiedKFold speed\nskf = StratifiedKFold(n_splits=N_FOLDS, shuffle=SHUFFLE, random_state=SEED)\n\nfolds = list(skf.split(\n    # we don't actually need `X` to produce indices, only `y`\n    X=np.zeros(len(strat_column)),\n    y=strat_column\n))")


# In[ ]:


# let's check whether class balance is preserved
print('train')
print(strat_column.iloc[folds[0][0]].value_counts(normalize=True).sort_index())
print('val')
print(strat_column.iloc[folds[0][1]].value_counts(normalize=True).sort_index())


# In[ ]:


# check all indices are there in joined validation blocks
assert len(set(np.hstack([v for (tr,v) in folds]).tolist())) == len(strat_column)


# In[ ]:


get_ipython().run_cell_magic('time', '', "# let's check updated StratifiedKFold speed\nskf = CustomStratifiedKFold(n_splits=N_FOLDS, shuffle=SHUFFLE, random_state=SEED)\n\nfolds = list(skf.split(\n    # we don't actually need `X` to produce indices, only `y`\n    X=None,\n    y=strat_column\n))")


#  Well, even on smaller dataset we get** 3x+** speed improvement

# In[ ]:


# let's check whether class balance is preserved also in new method
print('train')
print(strat_column.iloc[folds[1][0]].value_counts(normalize=True).sort_index())
print('val')
print(strat_column.iloc[folds[1][1]].value_counts(normalize=True).sort_index())


# In[ ]:


# check all indices are there in joined validation blocks (also for the new strategy)
assert len(set(np.hstack([v for (tr,v) in folds]).tolist())) == len(strat_column)


# ### let's create real-world example

# In[ ]:


N_FOLDS = 5
SEED = 42
SHUFFLE = True
NUM_UNIQUES = 10000
N = 20000000

strat_column = pd.Series(randomizer.randint(0, NUM_UNIQUES, N, dtype=np.int32))
print(strat_column.nunique())
strat_column.head(10)


# In[ ]:


get_ipython().run_cell_magic('time', '', "# let's check updated StratifiedKFold speed on heavier task (however, notice rapid memory peak)\n# don't try to run this with usual `StratifiedKFold` or prepare to wait A LOT\nskf = CustomStratifiedKFold(n_splits=N_FOLDS, shuffle=SHUFFLE, random_state=SEED)\n\nfolds = list(skf.split(\n    # we don't actually need `X` to produce indices, only `y`\n    X=None,\n    y=strat_column\n))")


# In[ ]:


# let's check whether class balance is preserved also in new method
print('train')
print(strat_column.iloc[folds[1][0]].value_counts(normalize=True).sort_index())
print('val')
print(strat_column.iloc[folds[1][1]].value_counts(normalize=True).sort_index())


# Well, almost as fast (or even faster at commiting mode) as `StratifiedKFold` on much smaller/less diverse dataset!

# In[ ]:


get_ipython().run_cell_magic('time', '', "skf = CustomStratifiedKFold2(n_splits=N_FOLDS, shuffle=SHUFFLE, random_state=SEED)\n\nfolds = list(skf.split(\n    # we don't actually need `X` to produce indices, only `y`\n    X=None,\n    y=strat_column\n))")


# In[ ]:


# let's check whether class balance is preserved also in new method
print('train')
print(strat_column.iloc[folds[1][0]].value_counts(normalize=True).sort_index())
print('val')
print(strat_column.iloc[folds[1][1]].value_counts(normalize=True).sort_index())


# In[ ]:




