#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import data visualization
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# import StratifiedKFold
from sklearn.model_selection import StratifiedKFold


# # Bengali GroupShuffleSplit
# 
# BengaliGroupShuffleSplit is a custom made validation strategy that random shuffles groups while making sure that graphemes in validation set may be constructed from root-consonant-vowel components present in training set.
# 
# As competition hosts clarified [here](https://www.kaggle.com/c/bengaliai-cv19/discussion/123002#707668) that test includes graphemes that are not present in the train but are made of train root-consonant-vowel components. Furthermore, in data decription hosts specify that ["the goal of competition is recognition of grapheme components rather than on recognizing whole graphemes."](https://www.kaggle.com/c/bengaliai-cv19/data)

# ## Original GroupShuffleSplit
# from [sklearn](https://scikit-learn.org/stable/modules/cross_validation.html)
# The GroupShuffleSplit iterator behaves as a combination of ShuffleSplit and LeavePGroupsOut, and generates a sequence of randomized partitions in which a subset of groups are held out for each split.
# 
# ![](https://scikit-learn.org/stable/_images/sphx_glr_plot_cv_indices_008.png)

# In[ ]:


# setup the input data folder
DATA_PATH = '../input/bengaliai-cv19/'


# In[ ]:


# load the dataframes with labels
train_labels = pd.read_csv(DATA_PATH + 'train.csv')
test_labels = pd.read_csv(DATA_PATH + 'test.csv')
class_map = pd.read_csv(DATA_PATH + 'class_map.csv')
sample_submission = pd.read_csv(DATA_PATH + 'sample_submission.csv')


# In[ ]:


NUM_FOLDS = 5


# ## Observation 1: Groups are well balanced.
# Every grapheme is a Root-Vowel-Consonant combination. Every combination corresponds to a group of roughly 150 images. So image groups are very well balanced; GroupShuffleSplit staregy should work well without rebalancing groups.

# In[ ]:


#Lets find total number of combinations:
combinations = train_labels.groupby(by=['vowel_diacritic','consonant_diacritic', 'grapheme_root'])            .count().reset_index().drop(['grapheme'],axis=1)
combinations.rename(columns={'image_id': 'image_count'}, inplace=True)
combinations.image_count.describe()


# ## Observation 2: Special case of Single Combination Roots
# GroupShuffleSplit cannot be applied to roots that are present in a single combination (*protected_roots*). If all such combinations are excluded from train, model wont be able to recognise the root coorrectly as it has never seen its label. Alternatively, if we exclude the combinations from validations, we will not know know how well model performs for these components.
# 
# For *protected_roots* we will use **graphemewise or rootwise** [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold), so that we have *protected_roots* both in train and validation set.
# 
# ![](https://scikit-learn.org/stable/_images/sphx_glr_plot_cv_indices_003.png)

# In[ ]:


#Add 'fold' column to train_labels set to store fold info:
train_labels['fold'] = np.nan


# In[ ]:


# Separate protected_roots:
img_roots = combinations.groupby(by=['grapheme_root']).count()
protected_roots = img_roots[img_roots.image_count == 1].index.values
protected_roots


# In[ ]:


#Rootwise StratifiedKFold for images containing protected_roots:
protected_imgs = train_labels[train_labels['grapheme_root'].isin(protected_roots)]['grapheme_root']

X = protected_imgs.index.values
y = protected_imgs.values
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True)
skf.get_n_splits(X, y)
print(skf)

i_fold = 0
for train_index, test_index in skf.split(X, y):
    train_labels.loc[X[test_index],'fold'] = i_fold
    i_fold += 1


# In[ ]:


train_labels.groupby(by=['fold']).count()['image_id']


# # BengaliGroupShuffleSplit
# A handmade *BengaliGroupShuffleSplit* is needed because GroupShuffleSplit does not check if combinations in validation set can be composed of components given in the train set.

# In[ ]:


# Drop combinations with protected_roots from combinations pool:
combinations = combinations[~combinations['grapheme_root'].isin(protected_roots)]
index_pool = set(combinations.index.values)


# In[ ]:


#Suppose we want to do 5 folds, so we need to save this many combinations for validation:
COMBS_PER_FOLD = combinations.shape[0]//NUM_FOLDS
COMBS_PER_FOLD


# In[ ]:


# Split all combinations by folds, while checking that all elements in validation set are present in train set.
import random
combinations['fold'] = 0
unused_idx = index_pool.copy()

for i in range(NUM_FOLDS):
    counter = COMBS_PER_FOLD
    valid_set = set()
    protected_set = set()
    loop_pool = index_pool.copy()
    while (counter and unused_idx):
        # get a random grapheme candidate from pool of candidate graphemes
        candidate = random.sample(unused_idx.intersection(loop_pool), 1)[0]
        loop_pool.remove(candidate)
        # check if all components of a candidate grapheme are present in the pool
        if (combinations.loc[loop_pool]['consonant_diacritic'].isin([combinations['consonant_diacritic'][candidate]]).any() and 
            combinations.loc[loop_pool]['vowel_diacritic'].isin([combinations['vowel_diacritic'][candidate]]).any() and
            combinations.loc[loop_pool]['grapheme_root'].isin([combinations['grapheme_root'][candidate]]).any()):
                
                # if TRUE add candidate grapheme to validation set
                valid_set.add(candidate)
                counter -= 1
        else:
            # if any of the components are missing from candidates pool, "leave it in training set"
            protected_set.add(candidate)
            
    if i==4: valid_set |= unused_idx
    unused_idx -= valid_set
    
    combinations.loc[valid_set,'fold'] = i

    print('Loop: {} Number of unused idx: {} Len protected_Set: {}'          .format(i, len(unused_idx), len(protected_set)))

#Slow. Got to be an elegant way to do this faster.


# In[ ]:


#Make sure folds have equal number of groups:
combinations.groupby(by=['fold']).count()['image_count']

## 3 extra combinations in fold 4 could have been randomly added to different folds.


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfor i in range(combinations.shape[0]):\n    i_fold = combinations.iloc[i,:]['fold']\n    i_root = combinations.iloc[i,:]['grapheme_root']\n    i_vowel = combinations.iloc[i,:]['vowel_diacritic']\n    i_consonant = combinations.iloc[i,:]['consonant_diacritic']\n\n    i_mask = (train_labels['grapheme_root'] == i_root) &\\\n            (train_labels['vowel_diacritic'] == i_vowel) &\\\n            (train_labels['consonant_diacritic'] == i_consonant) \n    \n    train_labels.loc[train_labels[i_mask].index,'fold'] = i_fold")


# In[ ]:


# take a look at folds
train_labels.groupby(by=['fold']).count()['image_id']


# In[ ]:


train_labels[train_labels['fold'].isna()].shape[0]


# In[ ]:


#output
train_labels['fold'] = train_labels['fold'].astype('int')
train_labels.to_csv('train_BengaliGroupShuffle.csv', index = False)

