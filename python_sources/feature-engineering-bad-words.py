#!/usr/bin/env python
# coding: utf-8

# ## Feature Engineering - Counting Occurence of Hateful Words
# 
# _By: Nick Brooks_
# 
# I believe that counting the occurence of hateful words may provide as useful feature to detect toxic comments. Fighting this stuff aint pretty.
# 
# My [Bad Bad Word Dataset](https://www.kaggle.com/nicapotato/bad-bad-words) has been legitimized in the [competition forum](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/50265) and may be used in the competition.

# In[ ]:


import time
import datetime
start = time.time()

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

print("Datasets Used:")
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv').fillna(' ')
test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv').fillna(' ')
badwords = pd.read_csv('../input/bad-bad-words/bad-words.csv', header=None).iloc[:,0].tolist()

print("Data Shape:")
print("Train Shape: {} Rows, {} Columns".format(*df.shape))
print("Test Shape: {} Rows, {} Columns".format(*test.shape))
print("Bad Words Shape: lenght {}".format(len(badwords)))
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train_text = df['comment_text']
test_text = test['comment_text']
all_text = pd.DataFrame(pd.concat([train_text, test_text]))
print("All Data Shape: {} Text Rows".format(all_text.shape))


# In[ ]:


# Glance
print(badwords[:10])
df.head()


# ## Feature Engineering
# 
# **Procedures applied to each row:**
# - badwordcount: Number of times a word from my bad word list shows up.
# - num_words: number of words, seperated by spaces.
# - num_chars: number of characters
# - normchar_badwwords: badwords count normalized by number of characters 
# - normword_badwwords: badwords count normalized by number of words 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'df["badwordcount"] = df[\'comment_text\'].apply(\n    lambda comment: sum(comment.count(w) for w in badwords))\ndf[\'num_words\'] = df[\'comment_text\'].apply(\n        lambda comment: len(comment.split()))\ndf[\'num_chars\'] = df[\'comment_text\'].apply(len)\ndf["normchar_badwords"] = df["badwordcount"]/df[\'num_chars\']\ndf["normword_badwords"] = df["badwordcount"]/df[\'num_words\']')


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
f, ax = plt.subplots(figsize= [10,7])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
ax.set_title("Correlation Matrix for Toxity and New Features")
plt.show()


# - **Normalized badwords by character count (normchar_badwords)** 
# - **Normalized badwords by word count (normword_badwords)**  <br>
# are strongly correlated with the toxicity variables.
# 
# ## Help and Contribute
# 
# Add potential sources of foul language to the discussion of my [Bad Bad Words](https://www.kaggle.com/nicapotato/bad-bad-words) Dataset to improve these features.

# In[ ]:




