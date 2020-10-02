#!/usr/bin/env python
# coding: utf-8

# # TL;DR
# The dataset is **TOO LARGE** for the Kaggle Notebook RAM to load at once.

# # 0. Preparation

# In[ ]:


import numpy as np
import pandas as pd
import gc
import json

from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


DIR = '../input/tensorflow2-question-answering/'
PATH_TRAIN = DIR + 'simplified-nq-train.jsonl'
PATH_TEST = DIR + 'simplified-nq-test.jsonl'


# ### 0-1. Number of samples in train & test dataset

# In[ ]:


get_ipython().system("wc -l '../input/tensorflow2-question-answering/simplified-nq-train.jsonl'")
get_ipython().system("wc -l '../input/tensorflow2-question-answering/simplified-nq-test.jsonl'")


# # 1. Load .jsonl file iteratively

# As we know, one of the most common way to convert .jsonl file into pd.DataFrame is  `pd.read_json(FILENAME, orient='records', lines=True)`:

# In[ ]:


df_test = pd.read_json(PATH_TEST, orient='records', lines=True)


# In[ ]:


df_test


# However, since we have a **HUGE train dataset** for this competition, Kaggle Notebook RAM cannot afford this method.
# Instead, we probablly have to load the train dataset iteratively:

# In[ ]:


json_train_head = []
N_HEAD = 10

with open(PATH_TRAIN, 'rt') as f:
    for i in range(N_HEAD):
        json_train_head.append(json.loads(f.readline()))


# In[ ]:


df_train_head = pd.DataFrame(json_train_head)


# In[ ]:


df_train_head


# In[ ]:


df_train_head.iloc[0,:].loc['long_answer_candidates']


# In[ ]:


df_train_head.iloc[0,:].loc['annotations']


# In[ ]:


del df_train_head, df_test
gc.collect()


# # 2. Data Visualization

# ## 2-1. Obtain data

# We must be cautious that **"short answer" for this competition corresponds to "yes-no answer" in the original dataset**.  

# In[ ]:


N_TRAIN = 307373
n_long_candidates_train = np.zeros(N_TRAIN)
t_long_train = np.zeros((N_TRAIN,2))
t_yesno_train = []


# In[ ]:


with open(PATH_TRAIN, 'rt') as f:
    for i in tqdm(range(N_TRAIN)):
        dic = json.loads(f.readline())
        n_long_candidates_train[i] = len(dic['long_answer_candidates'])
        t_long_train[i,0] = dic['annotations'][0]['long_answer']['start_token']
        t_long_train[i,1] = dic['annotations'][0]['long_answer']['end_token']
        t_yesno_train.append(dic['annotations'][0]['yes_no_answer'])


# In[ ]:


N_TEST = 345
n_long_candidates_test = np.zeros(N_TEST)


# In[ ]:


with open(PATH_TEST, 'rt') as f:
    for i in tqdm(range(N_TEST)):
        dic = json.loads(f.readline())
        n_long_candidates_test[i] = len(dic['long_answer_candidates'])


# ## 2-2. Visualization

# ### 2-2-1. Number of long answer candidates

# Some of data for long answers are swamped with a lot of candidates (**7946 in maximum!**):

# In[ ]:


pd.Series(n_long_candidates_train).describe()


# In[ ]:


pd.Series(n_long_candidates_test).describe()


# In[ ]:


plt.hist(n_long_candidates_train, bins=64, alpha=0.5, color='c')


# In[ ]:


plt.hist(n_long_candidates_train[n_long_candidates_train < np.max(n_long_candidates_test)], density=True, bins=64, alpha=0.5, color='c')
plt.hist(n_long_candidates_test, density=True, bins=64, alpha=0.5, color='orange')


# ### 2-2-2. Yes-no answer labels

# We can see significant class imbalance in yes-no answer labels.

# In[ ]:


plt.hist(t_yesno_train, bins=[0,1,2,3], align='left', density=True, rwidth=0.6, color='lightseagreen')


# ### 2-2-3. Long answer labels

# Description of start token labels:

# In[ ]:


pd.Series(t_long_train[:,0]).describe()


# Desciption of end token labels:

# In[ ]:


pd.Series(t_long_train[:,1]).describe()


# We can see below that nearly half of the long answers have start/end token -1.  
# In other words, there are a considerable number of '**NO ANSWERS**' in long answer labels, not only in yes-no labels:

# In[ ]:


print('{0:.1f}% of start tokens are -1.'.format(np.sum(t_long_train[:,0] < 0) / N_TRAIN * 100))
print('{0:.1f}% of end tokens are -1.'.format(np.sum(t_long_train[:,1] < 0) / N_TRAIN * 100))


# If the start token is -1, the corresponding end token is also -1:

# In[ ]:


np.sum(t_long_train[:,0] * t_long_train[:,1] < 0)


# The heatmap below tells us that:
# - when the start token and/or the end token are -1, yes-no answer is 'NONE'
# - yes-no answer 'NONE' does not always mean that the start token and/or the end token are -1

# In[ ]:


# no_answer_state[1,:] is the number of train data whose start token and end token are -1
# no_answer_state[:,1] is the number of train data whose yes-no answer is 'NONE'

no_answer_state = np.zeros((2,2))
no_answer_state[1,1] = np.sum((t_long_train[:,0]==-1) * (np.array([ 1 if t=='NONE' else 0 for t in t_yesno_train ])))
no_answer_state[1,0] = np.sum((t_long_train[:,0]==-1) * (np.array([ 0 if t=='NONE' else 1 for t in t_yesno_train ])))
no_answer_state[0,1] = np.sum((t_long_train[:,0]>=0) * (np.array([ 1 if t=='NONE' else 0 for t in t_yesno_train ])))
no_answer_state[0,0] = np.sum((t_long_train[:,0]>=0) * (np.array([ 0 if t=='NONE' else 1 for t in t_yesno_train ])))                             


# In[ ]:


no_answer_state


# In[ ]:


sns.heatmap(no_answer_state / N_TRAIN, annot=True, fmt='.3f', vmin=0, vmax=1, cmap='Blues_r')


# In[ ]:


del n_long_candidates_train, n_long_candidates_test, t_long_train, t_yesno_train, no_answer_state
gc.collect()


# ## 3. Text Word Counts

# Let us look into word counts of question texts & document texts.

# ### 3-1. Obtain data

# In[ ]:


q_lens_train = np.zeros(N_TRAIN)
d_lens_train = np.zeros(N_TRAIN)


# In[ ]:


with open(PATH_TRAIN, 'rt') as f:
    for i in tqdm(range(N_TRAIN)):
        dic = json.loads(f.readline())
        q_lens_train[i] = len(dic['question_text'].split())
        d_lens_train[i] = len(dic['document_text'].split())


# In[ ]:


q_lens_test = np.zeros(N_TEST)
d_lens_test = np.zeros(N_TEST)


# In[ ]:


with open(PATH_TEST, 'rt') as f:
    for i in tqdm(range(N_TEST)):
        dic = json.loads(f.readline())
        q_lens_test[i] = len(dic['question_text'].split())
        d_lens_test[i] = len(dic['document_text'].split())


# ### 3-2. Visualization

# #### 3-2-1. Word counts of question text

# In[ ]:


plt.hist(q_lens_train, density=True, bins=8, alpha=0.5, color='c')
plt.hist(q_lens_test, density=True, bins=8, alpha=0.5, color='orange')


# #### 3-2-2. Word counts of document text

# In[ ]:


plt.hist(d_lens_train, density=True, bins=64, alpha=0.5, color='c')
plt.hist(d_lens_test, density=True, bins=64, alpha=0.5, color='orange')


# Let us have fun!  
# Comments and recommendations will be welcomed ;)
