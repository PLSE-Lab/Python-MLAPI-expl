#!/usr/bin/env python
# coding: utf-8

# ## Why is my leaderboard score so low, my accuracy / F1 is so much better ?

# This seems to be a (beginner?) question here in the kernels/discussion and I had the same issue.
# Turns out it is quite simple:
# 
#  - your model sucks and you use the wrong metric! ;-) (applies to me too) 
#  - The evaluation metric for this competition (and therefore the Leaderboard) is **Macro**-F1
#  - Some (most?) libraries default to **Micro**-F1 and simple accuracy scores behave similarily
#  - Micro-F1 gives you much better score in this competition than Macro-F1, which is why your "local" score is better than the leaderboard
# 
# UPDATE: If you are sure you are already using the correct metric and still have low LB scores, do check out this discussion:
# https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/69366
# It basically points out the fact, that the **order of the image-ids in the submission file has to be identical with the sample_submission.csv**. So reorder your results accordingly!

# ## So what's the difference?

# The difference is mainly in when/where the averages are taken, and that makes a huge difference:
# - Micro-F1 basically adds up all the metrics (true positives, ...) accross classes and calculates f1 on the averages. Most accuracy scores do the same.
# - Macro-F1 first aggregates the classes in themselves (columns) before calculating the scores for each, then calculates the average
# 

# ## Show me some code!

# In[ ]:


import numpy as np
from sklearn import metrics 


# In[ ]:


n = 8 # number of 'training examples'

# create some dummy data 
y_pred = np.zeros(n*10).reshape((n, 10))
y_true = np.zeros(n*10).reshape((n, 10))
y_pred[:4] = [1,0,0,0,0,0,0,0,0,0] # (play with it to see the effects!)
y_true[:] = [1,1,0,0,0,0,0,0,0,0] # (play with it to see the effects!)


# In[ ]:


print('Micro F1:',metrics.f1_score(y_true, y_pred, average='micro'))
print('Macro F1:',metrics.f1_score(y_true, y_pred, average='macro')) 


# In[ ]:


# Let's recreate the functions and have a closer look:

def f1_micro(y_true, y_preds, thresh=0.5, eps=1e-20):
    preds_bin = y_preds > thresh # binary representation from probabilities (not relevant)
    truepos = preds_bin * y_true
    
    p = truepos.sum() / (preds_bin.sum() + eps) # take sums and calculate precision on scalars
    r = truepos.sum() / (y_true.sum() + eps) # take sums and calculate recall on scalars
    
    f1 = 2*p*r / (p+r+eps) # we calculate f1 on scalars
    return f1

def f1_macro(y_true, y_preds, thresh=0.5, eps=1e-20):
    preds_bin = y_preds > thresh # binary representation from probabilities (not relevant)
    truepos = preds_bin * y_true

    p = truepos.sum(axis=0) / (preds_bin.sum(axis=0) + eps) # sum along axis=0 (classes)
                                                            # and calculate precision array
    r = truepos.sum(axis=0) / (y_true.sum(axis=0) + eps)    # sum along axis=0 (classes) 
                                                            #  and calculate recall array

    f1 = 2*p*r / (p+r+eps) # we calculate f1 on arrays
    return np.mean(f1) # we take the average of the individual f1 scores at the very end!

print('Micro F1 (sklearn):',metrics.f1_score(y_true, y_pred, average='micro'))
print('Micro F1 (own)    :',f1_micro(y_true, y_pred))
print('Macro F1 (sklearn):',metrics.f1_score(y_true, y_pred, average='macro')) 
print('Macro F1 (own)    :',f1_macro(y_true, y_pred))


# Obviously, those functions can be combined into one , use `axis=None` to generate micro, calculate the mean always. They were separated for educational purposes only.
# 
# Of course, this doesn't help you get a better score, but it should help you iterate faster, when your score actually reflects the leaderboard better. ;-)
# Good Luck!
