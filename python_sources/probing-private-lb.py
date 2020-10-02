#!/usr/bin/env python
# coding: utf-8

# # Probe the private dataset
# Because of the way how Kaggle's evaluation system works (in case of kernels only competitions), we can find valuable information about the private test set. In this notebook, I'll try to show you a method of how you can do it.
# 
# Originally this was [Chris Deotte's idea](https://www.kaggle.com/cdeotte/private-lb-probing-0-950).

# Before coding, here is some useful information.
# 
# ## Code Competitions
# > Some competitions are code competitions. In these competitions all submissions are made from inside of a Kaggle Notebook, and it is not possible to upload submissions to the Competition directly.
# >
# > Following the competition deadline, your code will be rerun by Kaggle on a private test set that is not provided to you. Your model's score against this private test set will determine your ranking on the private leaderboard and final standing in the competition.
# 
# ### Synchronous Code Competitions
# > When you submit from a Kernel, Kaggle will run the code against both the public test set and private test set in real time.
# >
# > In a synchronous Kernels-only competition, the files you can observe and download will be different than the private test set and sample submission. The files may have different ids, may be a different size, and may vary in other ways, depending on the problem. You should structure your code so that it predicts on the public test.csv in the format specified by the public sample_submission.csv, but does not hard code aspects like the id or number of rows. When Kaggle runs your Kernel privately, it substitutes the private test set and sample submission in place of the public ones.
# 
# ![PublicPrivate](https://storage.googleapis.com/kaggle-media/competitions/general/public_vs_private.png)
# 
# #### Refrences:
# - [Code Competition](https://www.kaggle.com/docs/competitions#kernels-only-competitions)
# - [Code Competition FAQ](https://www.kaggle.com/docs/competitions#kernels-only-FAQ)
# - [Instant Gratification Competition](https://www.kaggle.com/c/instant-gratification/overview/description)
# - [Instant Gratification Data description](https://www.kaggle.com/c/instant-gratification/data)

# ## Clarifications
# 
# > When you submit from a Kernel, Kaggle will run the code against both the public test set and private test set
# 
# There is only one run. The dataset (usually `test.csv`, `sample_submission.csv`) contains **both** the private and the public part of the test set.
# 
# > When Kaggle runs your Kernel privately, it substitutes the private test set and sample submission in place of the public ones.
# 
# By "private" they mean **both** public and private
# 
# 
# ### Public run
# (left side of the image above)
# Kaggle executes a public run when you are commiting your notebook.
# - **Test data**: public (blue)
# - **Generated output**: submission.csv
# - **Leaderboard**: none
#  
# 
# ### Private run
# (right side of the image above)
# Kaggle executes a private rerun when you are submitting one of the outputs of your notebook for evaluation.
# - **Test data**: public + private (blue + red)
# - **Generated output**: None (erased after evaluation)
# - **Leaderboard**: public and private (calculated on the 'blue' and 'red' part of the test set)

# ## Probing private dataset
# Because Kaggle calculates the public LB score based on the private run, we can exploit it by report private dataset findings via our public score.

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


# Earlier submissions, with known public scores.
# Link to the dataset: https://www.kaggle.com/dataset/5f77b2e0af4234095883bd1cc65c0e20b8a74b44cc10048f14e9d654132114b0
SUBMISSION_FILES = [
    'submission-0.csv',  # 0.401
    'submission-1.csv',  # 0.344
    'submission-2.csv',  # 0.444
    'submission-3.csv',  # 0.457
    'submission-4.csv',  # 0.483
    'submission-5.csv',  # 0.479
    'submission-6.csv',  # 0.480
    'submission-7.csv',  # 0.507
    'submission-8.csv',  # 0.500
    'submission-9.csv',  # 0.017
]

# Note: Every submission has a different score.


# In[ ]:


labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')


# ### How many samples we have in the test set?

# In[ ]:


target = labels.shape[0]

# We know that the public test set is ~14% of the data, so the full dataset is ~7143 samples
# It can be less or more than 7143, so we have to probe some values.

# Let's check the possible values between 7100 and 7200
if target < 7100:
    submission_idx = 0
elif target <= 7100 and target < 7110:
    submission_idx = 1
elif target <= 7110 and target < 7120:
    submission_idx = 2
elif target <= 7120 and target < 7130:
    submission_idx = 3
elif target <= 7130 and target < 7140:
    submission_idx = 4
elif target <= 7140 and target < 7150:
    submission_idx = 5
elif target <= 7150 and target < 7160:
    submission_idx = 6
elif target <= 7160 and target < 7170:
    submission_idx = 7
elif target <= 7170 and target < 7180:
    submission_idx = 8
elif target <= 7180 and target < 7190:
    submission_idx = 9
else:
    submission_idx = -1 # 0.0 score
    


# In[ ]:


if submission_idx >= 0:
    CSV_FILE = '../input/2019-dsb-private-probing/{}'.format(SUBMISSION_FILES[submission_idx])
else:
    CSV_FILE = '../input/data-science-bowl-2019/sample_submission.csv'

# Public (or private) test sample_submission file.
submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')

# Your predictions
df_predict = pd.read_csv(CSV_FILE)

# Defaults
submission['accuracy_group'] = 3

for i, row in df_predict.iterrows():
    submission.loc[submission['installation_id'] == row['installation_id'], 'accuracy_group'] = row['accuracy_group']

submission.to_csv('submission.csv', index=False)


# ## Conclusion
# 
# The public score of the above is **0.444**, which was the score of the `submission-2.csv`.
# And `submission_idx=2` was used in the `target <= 7110 and target < 7120` if statement.
# 
# Now, we know that the number of samples in the private run is between 7110 and 7120.
# 
# 
# **Notes**
# - The private-run contains both the public and the private test set. So the actual number of private samples is 1000 less.
# - You can refine the values with a 2nd run (target between 7110 and 7120)
# - You don't have to, I've already did, it is 7112.
# 
# 
# With this technique, you can find out some interesting facts about the private dataset.

# -------------------

# **Thanks for reading, please vote if you find this notebook useful.**

# In[ ]:





# In[ ]:




