#!/usr/bin/env python
# coding: utf-8

# Hi guys!
# <br>I do not know about the scale of the problem, but personally, when I tried to setup validation strategy with 5+ folds and **1000+ **unique groups for this particular dataset of **20+M rows** I stuck with usual `sklearn`-based `StratifiedKFold` - it runs incredibly long to actually return train/val indices
# <br>That's why I had to spend some time to reduce the speed bottleneck (at the cost of less efficient memory usage)
# <br>In this small kernel I want to share with you faster stratified cross-validator built from scratch.
# <br>**P.s.** small speed tests are also provided. 
# <br>**UPD1**: Made it more-or-less generator-like to reduce RAM usage on final stage
# <br>**UPD2**: corrected indexes, now is fully stratified (class balance is preserved as in `StratifiedKFold`)

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

# **P.s.** There are a lot that needs polishing in current approach - and one may optimize my script to reduce memory peaks, further improve speed etc. However, it works (almost) as intended to be :)
# <br>Hope you guys found this code useful
# <br>Comments, likes, new ideas are highly welcomed!
# <br>Happy kaggling!
# 
# ---
# Check my latest notebooks:
# - [Aligning Temperature Timestamp](https://www.kaggle.com/frednavruzov/aligning-temperature-timestamp)
# - [NaN restoration techniques for weather data](https://www.kaggle.com/frednavruzov/nan-restoration-techniques-for-weather-data/edit/run/22556654)
# 
