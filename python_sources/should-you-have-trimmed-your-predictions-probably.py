#!/usr/bin/env python
# coding: utf-8

# Given the number of upsets this year, there has been some discussions ([[1]](https://www.kaggle.com/c/mens-machine-learning-competition-2018/discussion/52184), [[2]](https://www.kaggle.com/c/mens-machine-learning-competition-2018/discussion/52208))  of *trimming* predictions in order to minimize our log loss in case of an upset. Here is a simple notebook that explores this idea a bit further. 
# 
# To be precise, for a trim amount $0\leq\alpha\leq 1$ we specify an interval $I=[\alpha, 1-\alpha]$ and we will reshape our predictions to be in the interval $I$. The idea is to be a conservative gambler and keep our predictions away from zero and one because the log loss will punish us if we are wrong. We want to find the best $\alpha$ which will minimize the log loss of the truth and our trimmed predictions. 
# 
# First we load the data. I am using my submitted predictions for the competition which are from a logistic regression model. I'm also using the actual results of the first 60 games of the 2018 tournament; this data was obtained from [Sports-Reference](https://www.sports-reference.com/cbb/play-index/tourney.cgi). 
# 
# 
# 

# In[1]:


import pandas as pd 
import numpy as np
import os

import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

def trim_pred(x, alpha):
    upper = 1-alpha
    lower = alpha
    if x > upper:
        return upper
    if x < lower:
        return lower
    else: return x
    
def trimmed_loss(alpha):
    trimmed_preds = [trim_pred(x,alpha) for x in data.Pred]
    return log_loss(data.Result , trimmed_preds)

def annot_min(x,y, ax=None):
    minIxVal = np.argmin(y);
    zeroBasedIx = y[minIxVal];
    xmin = x[minIxVal];
    ymin = y[minIxVal]
    text = "Minimum: Trim Interval = [{}, {}], Log Loss = {}".format(round(xmin,2), round(1-xmin,2), round(ymin, 3))
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="round,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=0.1")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmin, ymin), xytext=(0.94,0.90), **kw)
    

path = "../input/ncaa-2018-preds-and-truth/"

preds = pd.read_csv(os.path.join(path,"2018_predictions_logistic.csv")) 
truth = pd.read_csv(os.path.join(path,"truth.csv")) 

data = truth.merge(preds, left_on='ID', right_on='ID', how='inner')
data.head()


# Now we plot the log loss of the trimmed predictions against the truth in order to see how effective a trim would be.
# 
# We search for $\alpha$ in the interval $[0, 1/4]$ to find the ideal $\alpha$.  The function ```trimmed_loss(alpha)``` computes the log loss of the trimmed predictions against the truth. The minimum value is labeled in the plot and we see that some amount of trim would have benefited me in this competition, but too much trim is detrimental. The amount we shave off is actually minimal, but this can still lead to big movements on the leaderboard. 
# 
# This observation does not help us much now that the competition has started, but it is something to keep in mind in the future. In particular, would it have been helpful to find the trim quantity $\alpha$ in a data-driven manner? (For example by treating it as a parameter and tuning it with the previous tournaments.)
# 
# Try this out with your own predictions and let me know what you think!

# In[2]:


xvals = np.arange(0, .30, 0.001)
yvals = [trimmed_loss(alpha) for alpha in xvals]

plt.figure(figsize=(10,10))
plt.plot(xvals, yvals)
plt.xlabel('Alpha (Trim Amount)', fontsize=14)
plt.ylabel('Log Loss', fontsize=14)
annot_min(list(xvals),yvals, ax=None)
plt.show()


# In[ ]:




