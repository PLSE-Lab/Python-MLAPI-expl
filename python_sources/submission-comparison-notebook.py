#!/usr/bin/env python
# coding: utf-8

# This notebook compares the results of two popular kernels ([NB-SVM strong linear baseline](https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline/output) and [Improved LSTM baseline](https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout/output)). Examining the outliers can show which comments the models disagree strongly on. I don't expect this to be used for any serious model adjustments, but it can give clues towards additional preprocessing which may be helpful. 
# 
# And even if it is not helpful, it's still interesting to see which comments your models do not agree on. I'm sure other people have better ways of performing this task too - please leave a comment! 

# In[21]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Load the data: 

# In[22]:


pd.options.display.max_colwidth = 500

sub1 = pd.read_csv('../input/nb-svm-strong-linear-baseline/submission.csv')
sub2 = pd.read_csv('../input/improved-lstm-baseline-glove-dropout/submission.csv')

train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')


# Subtract the submission scores and square the result: 

# In[23]:


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
res_compare = sub1.copy()
res_compare[label_cols] = (sub1[label_cols] - sub2[label_cols])**2
res_compare['diff'] = res_compare.sum(axis=1)


# Add the comments themselves: 

# In[24]:


res_compare['comment_text'] = test['comment_text']


# Order the results and show the worst offenders at the top: 

# In[32]:


res_compare.sort_values('diff', ascending=False).loc[:,['id','comment_text','diff']]


# We can see that at the top we have some comments which are most certainly toxic! But for some reason, they're being assessed differently by the different models. At the bottom, we can see those generic "template" comments which the models can both agree are not toxic whatsoever. 

# Let's take a look at an example of the top comment we identified: 

# In[16]:


sub1[sub1.id=='76cb5742586f4c2e'] #NB-SVM


# In[17]:


sub2[sub2.id=='76cb5742586f4c2e'] #LSTM


# In[18]:


test[test.id=='76cb5742586f4c2e'] #reminder of the comment


# So the NB-SVM model seems, in this case, to better predict the toxicity of this comment! 

# Let's save the results for further analysis offline: 

# In[20]:


res_compare.to_csv('comparison.csv')


# In[ ]:




