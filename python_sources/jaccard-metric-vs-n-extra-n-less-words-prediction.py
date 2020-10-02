#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from ipywidgets import interactive
import matplotlib.pyplot as plt


# ## In this notebook we analyse the relationship between the jaccard metric and length of the predicted text. I have started analysis to understand impact of a case, if I predict one extra word vs one word less compared to ground truth how would jaccard metric behave across different lengths. Will delve into insights at end of notebook

# #### Read data

# In[ ]:


train_data_set = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
train_data_set.head()


# ### Understand length of words in tweet text and selected_text columns

# In[ ]:


train_data_set['tweet_length'] = train_data_set['text'].apply(lambda x: len(str(x).split()))

train_data_set['senti_length'] = train_data_set['selected_text'].apply(lambda x: len(str(x).split()))


# Mean sentiment length text is around 7 and mean tweet length is around 12 as shown in describe below

# In[ ]:


train_data_set.describe()


# In[ ]:


def jac(gt_len,pred_len):
    return float(min(gt_len,pred_len)) / (gt_len+pred_len - (min(gt_len,pred_len)))


# ### Generating data 
# - Assuming that there is atleast 1 word match between ground truth and prediction, since below this anyway jaccard will be 0 so no point in analysing it across lengths
# - Example: From a orginal tweet text of 33 words we can get predictions in range of 1-33, and selected text(ground truth) can be in the same range 1-33
# - So looking at distrubution in train data tweet text has word count range between 1-33, generating data below accordingly

# In[ ]:


gt_len = list(range(1,train_data_set['tweet_length'].max()))
gt_len = gt_len*len(gt_len)

pred_len = gt_len.copy()
pred_len.sort()

cases_df = pd.DataFrame({'ground_truth':gt_len,'pred_len':pred_len})

cases_df['jaccard_metric'] = cases_df.apply(lambda x: jac(x['ground_truth'],x['pred_len']),axis=1)

cases_df['under_or_over_pred'] = cases_df['pred_len'] - cases_df['ground_truth']

cases_df.sort_values(by=['ground_truth','pred_len'],inplace=True)


# ### Analysis: When the selected text/ ground truth has 4 words, how can prediction length vary jaccard
# - For example at median ground truth length 4 if we predict one extra word (overpredict) jaccard is at 0.8 whereas if we predict one word less(underpredict) jaccard is at 0.75 and the rate of decrease in jaccard if we underpredict is 0.25 and linear whereas for overpredict its not linear which can be seen from the plot in next cells

# In[ ]:


cases_df[cases_df['ground_truth']==4].head(10)


# ### Observe the steepness of curve in the left hand side(under prediction region) vs the right hand side(over prediction region)

# In[ ]:


def f(gt):
    cases_df2 = cases_df[cases_df.ground_truth==gt]
    plt.plot(cases_df2['under_or_over_pred'],cases_df2['jaccard_metric'],marker='o',linestyle='dashed')
    plt.xlabel('<--- under_pred___over_pred -->')
    plt.ylabel('jaccard')
    plt.show()

interactive_plot = interactive(f,gt=(1,33))
output = interactive_plot.children[-1]
interactive_plot


# ## Analysis conclusions:
# - Over prediction by a word is safer than underprediction unless the ground truth is not one word
# - As the length of ground truth sentence increases the impact of predicting one extra word vs one lesser decreases

# In[ ]:




