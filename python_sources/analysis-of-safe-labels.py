#!/usr/bin/env python
# coding: utf-8

# # Analysis of Safe Labels
# In this [forum thread](https://www.kaggle.com/c/melbourne-university-seizure-prediction/forums/t/25089/new-test-set-is-coming) the competition admins describe the file `train_and_test_data_labels_safe.csv` which contains "safe" training labels.
# 
# Here we compute some statistics to understand how complete that file is.

# In[ ]:


import pandas as pd


# In[ ]:


def check_new_files(t):
    t['patient'] = t['image'].apply(lambda x: x.split('_')[0])
    t['dataset'] = t['image'].apply(lambda x: 'old_train' if len(x.split('_'))==3 else 'old_test')
    stats = t.groupby(['patient', 'dataset', 'class', 'safe']).size().reset_index()
    stats.rename(columns={0: 'count'}, inplace=True)
    return stats


# In[ ]:


try:
    t = pd.read_csv('../input/train_and_test_data_labels_safe.csv')
    stats = check_new_files(t)
    stats
except OSError as e:
    print('Safe labels file not found')
    print(e)
    print('Screen shot of safe label analysis is below')


# ![Output of stats][1]
# 
# 
#   [1]: http://singsoftnext.com/safe_stats.png

# ## Discussion
# 
#   1. All training files in the originally dataset have been accounted for.
#   1. All training files originally labeled as `preictal` (class 1) are safe to use to train solutions.
#   1. Quite a few training files originally labeled as `interictal` are now labelled `unsafe` and should not be used. For example, for patient 1, 582 of the original `interictal` training files should be discarded. The remaining 570 files can continue to be used. (The number of training files in the original dataset is 1152 which equals 582 + 570, so all are accounted for.)
#   1. Several of the test files in the original dataset are now labeled as `preictal` and `safe` and so presumably could be used as training data for new solutions.
#   1. One might assume that any file from the original test set that is _not_ labeled `safe` and `preictal` is `interictal`. However, this has not been specifically stated. In any case, no `safe` labels are provided for those files. Some competitors have discovered experimentally that some of them are `unsafe`. So these files cannot safely be used to train solutions, based on the information provided.
