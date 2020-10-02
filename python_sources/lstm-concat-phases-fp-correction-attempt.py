#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

print(os.listdir('../input'))
print(os.listdir('../input/phase-subs'))

data_dir = '../input'


# ## Phase Fault Distributions and Concatenating Phases Problem
# 
# As we can see below, 57 of the lines have a fault in only 1 phase, and another 57 of the lines only have a fault in 2 phases. If the test set has a similar distribution of phase faults then a model such as https://www.kaggle.com/braquino/5-fold-lstm-attention-fully-commented-0-694 that concatenates all 3 phases and always predicts a fault for all 3 phases in a measurement then it will inherently predict some false positives. Concatenating the phases does seem to be very effective but maybe we can combine predictions from a model that trains and predicts on each phase individually to further improve the score.

# In[ ]:


metadata_train = pd.read_csv(data_dir + '/vsb-power-line-fault-detection/metadata_train.csv')


# In[ ]:


metadata_train['target_phase_sum'] = metadata_train.groupby('id_measurement')['target'].transform(np.sum)
metadata_train.head()


# In[ ]:


plt.hist(metadata_train['target_phase_sum'].values)
plt.show()


# In[ ]:


print(len(metadata_train[metadata_train['target_phase_sum'] == 1]))
print(len(metadata_train[metadata_train['target_phase_sum'] == 2]))
print(len(metadata_train[metadata_train['target_phase_sum'] == 3]))


# In[ ]:


metadata_train[metadata_train['target_phase_sum'] == 1].head(10)


# In[ ]:


metadata_train[metadata_train['target_phase_sum'] == 2].head(10)


# ## Compare Predicted Targets

# In[ ]:


# Submission from LSTM model trained on each phase seperately
# Source https://www.kaggle.com/braquino/5-fold-lstm-attention-fully-commented-0-694 with modification to train and predict on each phase
sub_phase = pd.read_csv(data_dir + '/phase-subs/lstm_5fold_phase_564_sub.csv')
sub_phase_n_faults = sub_phase.target.sum()
print(sub_phase_n_faults)


# In[ ]:


# Submission from LSTM model trained on concatenated phases
# Source https://www.kaggle.com/braquino/5-fold-lstm-attention-fully-commented-0-694
sub_phase_concat = pd.read_csv(data_dir + '/phase-subs/lstm_5fold_phase_concat_648_sub.csv')
sub_phase_concat_n_faults = sub_phase_concat.target.sum()
print(sub_phase_concat_n_faults)


# In[ ]:


diff_faults = sub_phase_n_faults - sub_phase_concat_n_faults
print(diff_faults)


# The phase only model predicted 239 more faults than the phase concat model, it scores lower on the LB so this might indicate it is predicting more false positives.

# In[ ]:


meta_test = pd.read_csv(data_dir + '/vsb-power-line-fault-detection/metadata_test.csv')


# In[ ]:


# Merge meta test meta data so we can compare predictions from each submission
sub_merge = sub_phase.copy()
sub_merge = sub_merge.drop(columns=['target'])
sub_merge['id_measurement'] = meta_test.id_measurement.values
sub_merge['phase'] = meta_test.phase.values
sub_merge['target_sub_phase'] = sub_phase['target'].values
sub_merge['target_sub_phase_concat'] = sub_phase_concat['target'].values
sub_merge.head()


# In[ ]:


sub_merge[sub_merge['target_sub_phase_concat'] == 1].head(50)


# The phase concat model contains many of the same predictions as the phase model, but notice target_sub_phase=0 for signal_id 9166, 9467 and 10059. I am thinking the phase concat model has predicted a false postive here. If the test set target distribtion is similar to the train set then the phase concat model would certainly predict some false postives since some faults only affect 1 or 2 phases.

# ## Blend Submissions

# Let's try improve the phase concat model submission by setting the suspected false positives to zero. 

# In[ ]:


sub_merge['target_sub_phase_group_sum'] = sub_merge.groupby('id_measurement')['target_sub_phase'].transform(np.sum)
sub_merge[sub_merge['target_sub_phase_concat'] == 1].head(20)


# In[ ]:


sub_merge['target'] = sub_merge.target_sub_phase_concat.values
sub_merge.loc[(sub_merge['target_sub_phase'] == 0) & (sub_merge['target_sub_phase_concat'] == 1) & (sub_merge['target_sub_phase_group_sum'] == 2), 'target'] = 0


# In[ ]:


sub_merge.head()


# In[ ]:


final_sub = sub_merge[['signal_id', 'target']]
final_sub.head()


# In[ ]:


final_sub.to_csv('submission.csv', index=False)


# The new submission scores 0.628 on the LB, so it's not an improvement on the phase concat model which scored 0.648. Maybe with a better scoring submissions from a model trained on each phase this method might help, or maybe every fault in the test set happens on all 3 phases. 
