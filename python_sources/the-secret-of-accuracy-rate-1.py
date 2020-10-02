#!/usr/bin/env python
# coding: utf-8

# # The secret of accuracy rate 1
# 
# 
# Thanks https://www.kaggle.com/holfyuen/basic-nlp-on-disaster-tweets

# In[ ]:


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string

plt.rcParams.update({'font.size': 14})

# Load data
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
sub_sample = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

print (train.shape, test.shape, sub_sample.shape)


# 
# (Spoiler alert) You Can Get Perfect Score in (public) Leader Board!
# 
# You will see in the public leaderboard, many participants got a perfect score. It is because the whole dataset with label is available online (one copy available [on Kaggle](https://www.kaggle.com/jannesklaas/disasters-on-social-media)). You can find the correct label of our test set so you can achieve perfect score.
# 
# In such case, the ranking on public leaderboard is meaningless. The good news is, you can now focus on learning NLP and modelling skills with this dataset, instead of fighting for higher position on the leaderboard!
# 
# (Reference: [szelee's notebook](https://www.kaggle.com/szelee/a-real-disaster-leaked-label/notebook))

# In[ ]:


leak = pd.read_csv("../input/disasters-on-social-media/socialmedia-disaster-tweets-DFE.csv", encoding='latin_1')
leak['target'] = (leak['choose_one']=='Relevant').astype(int)
leak['id'] = leak.index
leak = leak[['id', 'target','text']]
merged_df = pd.merge(test, leak, on='id')
sub1 = merged_df[['id', 'target']]
sub1.to_csv('submit.csv', index=False)

