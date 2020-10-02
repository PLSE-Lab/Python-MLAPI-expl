#!/usr/bin/env python
# coding: utf-8

# ### GAP metric
# The metric for this competition, [global average precision](https://www.kaggle.com/c/landmark-recognition-challenge#evaluation), is not one of the standard evaluation metrics in, for example, scikit-learn. Here is the code that I am using to compute it. Feel free to use it if you like. And if you see any errors, please comment, and we'll get them fixed up
# 
# 

# In[5]:


import numpy as np
import pandas as pd

def GAP_vector(pred, conf, true, return_x=False):
    '''
    Compute Global Average Precision (aka micro AP), the metric for the
    Google Landmark Recognition competition. 
    This function takes predictions, labels and confidence scores as vectors.
    In both predictions and ground-truth, use None/np.nan for "no label".

    Args:
        pred: vector of integer-coded predictions
        conf: vector of probability or confidence scores for pred
        true: vector of integer-coded labels for ground truth
        return_x: also return the data frame used in the calculation

    Returns:
        GAP score
    '''
    x = pd.DataFrame({'pred': pred, 'conf': conf, 'true': true})
    x.sort_values('conf', ascending=False, inplace=True, na_position='last')
    x['correct'] = (x.true == x.pred).astype(int)
    x['prec_k'] = x.correct.cumsum() / (np.arange(len(x)) + 1)
    x['term'] = x.prec_k * x.correct
    gap = x.term.sum() / x.true.count()
    if return_x:
        return gap, x
    else:
        return gap


# In[8]:


# Generate some random predictions on 3 classes
ypred = np.random.choice([1,2,3], 10)
ytrue = np.random.choice([1,2,3], 10)
conf = np.random.random(10)
gap, x = GAP_vector(ypred, conf, ytrue, True)
gap


# In[9]:


x


# In the data frame, correct is relevance, prec_k is the precision at rank k and term is the entire term under the summation in the formula:
# $$ \frac{1}{M}\sum_{i=1}^{N}Pr(i) rel(i)$$

# In[ ]:




