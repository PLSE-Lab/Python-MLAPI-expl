#!/usr/bin/env python
# coding: utf-8

# # Summary
# **This notebook illustrates the relation between the AUC score and the probability (p_hit) with which your model correctly identifies a download-click. It shall provide some intuition.**
# 
# **In particular, to achieve *0.9552*, we need a probability of *p_hit = 2.7%*  (that our predictions of download-clicks are correct) for the very first clicks, that we predict as download-clicks (the ones we are most sure of)**. 
# 
# This probability can and must fall to *p_hit = 0.24%* as we predict more and more clicks as download-clicks. Eventually we will and must predict that all clicks are download-clicks. 0.9552 currently corresponds to #2295 on the public leaderboard. To get 0.9827, which corresponds to #1 on the current public leaderboard, you would need to increase *p_hit = 2.7%* to *p_hit = 6.5%*
# 
# Confusion potential: *p_hit* is different from the probability *p_click* that we assign to each click (the probability, that it is a download-click). *p_hit* is the observed hit-rate after we have assigned a probability to each click, and then fixed a threshold *p_threshold*, such that we predict that each click where *p_click >= p_threshold*, is a download-click.
# 
# **Note: The numbers in this code are only illustrative. I didn't run any analysis on the actual training or test data. **
# 
# 
# # Factors to increase p_hit
# The minimum rate of correct predictions should be around *p_hit = 0.24%* if we predict randomly (because in the training set around 0.24% of all clicks are download-clicks). 
# 
# Suppose that half of the clicks are fraudulent (=clicks which were made fraudulently and will surely not become downloads), and we could identify them all correctly. Then, we would just take the remainig clicks, and randomly pick any number and predict that they are download-clicks. This would yield a probability around *p_hit = 0.48%* ( = 2 * 0.24%). 
# 
# The factors to increase *p_hit* are:
# * percentage of fraudulent clicks, which will surely not lead to a download
# * percentage of how many of the above fraudulent clicks we can identifiy (obviously, we will be able to identify some of those fraudulent clicks very reliably, and some others with less and less probability)
# * identify certain clicks among non-fraudulent clicks, which will yield a download more often than average (note: here we do not look at all at fraudulent clicks!)
# 
# 
# # How to plot the ROC curve
# Let R be the ROC curve. If (x,y) is an element of the set R, then
# * x = true positive rate = % (number of clicks correctly identified as leading to downloads / number of all clicks leading to downloads)
# * y = false positive rate = % (number of clicks falsely identified as leading to downloads / number of all clicks, which do NOT lead to downloads)
# 
# The following statements follow from the definition of the ROC curve:
# * The ROC curve starts at the point (x,y) = (0,0), thus at the beginning we must predict no clicks as downloads-clicks
# * The ROC curve ends at the point (x,y) = (1,1), thus at the end we have predicted that every click becomes an install, which leads us to have found every download-click (y=1), but at the same time we have predicted also every non-download-click as leading to a download (thus x=1)
# 
# Let us start by fixing the assumed number of clicks:

# In[2]:


number_non_download_clicks = 17960000  # illustrative number
number_download_clicks = 0.0024 * number_non_download_clicks  # 0.0024 corresponds to 0.24% as observed in the training set
number_all_clicks = number_non_download_clicks + number_download_clicks


# We now fix the number *n(p)* of clicks, that we identify as download-clicks. The number will depend on the threshold *p = p_threshold*. Note that we must have *n(1) = 0* and *n(0) = number_all_clicks*.  At the same time, we define the targeted probability *p_hit* that whatever we predict as a download-click will indeed be one:

# In[18]:


def n(p): return (1-p**4)*number_all_clicks
def p_hit(p):
    factor = 10  # use 10 for 0.9552 and 26 for 0.9827
    prob_to_get_hit = number_download_clicks*(1+factor*p)/number_all_clicks  # this formula is an assumption, NOT a derivation
    max_prob = number_download_clicks/n(p) if n(p)>0 else 0
    prob_to_get_hit = min(prob_to_get_hit, max_prob)  # cant have too high a probability because otherwise will predict more correct ones as there are
    return prob_to_get_hit

# calculate x and y of ROC curve, for each probability threshold p = p_threshold
def x(p): return n(p) * (1-p_hit(p)) / number_non_download_clicks
def y(p): return n(p) * p_hit(p) / number_download_clicks


# We proceed to plot the ROC-curve, by letting *p=p_threshold* run from 0 to 1. We also plot the graph for the number of clicks *n(p)*, that we predict as download-clicks, and the plot for the probability *p_hit(p)* that those are correct:

# In[19]:


import numpy as np
import matplotlib.pyplot as plt

delta = 0.0001
x_values = []
y_values = []
area_under_the_roc_curve = 0
x_equidistant = []
for p in np.arange(0.0, 1.0+delta, delta):
    x_equidistant.append(p)
    x_values.append(x(p))
    y_values.append(y(p))
    area_under_the_roc_curve += y(p) * (x(p) - x(p+delta))
x_equidistant_rev = x_equidistant[::-1]


plt.figure(figsize=(22,10))

plt.subplot(131)
plt.title('ROC-curve (area underneath is {:5.4f})'.format(area_under_the_roc_curve))
plt.scatter(x_values, y_values, marker='.')
plt.xlabel('x from 0 to 1 (as p goes from 1 to 0 !)')

plt.subplot(132)
plt.title('Probability p_hit that click is download-click')
plt.scatter(x_equidistant_rev, [p_hit(p) for p in x_equidistant_rev], marker='.')
plt.xlabel('p from 1 to 0')
plt.xlim([1, 0])

plt.subplot(133)
plt.title('Number of clicks predicted as download-clicks')
plt.scatter(x_equidistant_rev, [n(p) for p in x_equidistant_rev], marker='.')
plt.xlabel('p from 1 to 0')
plt.xlim([1, 0])

plt.show()


# Please let me know if you have any questions or comments. Corrections and other explanations are more than welcome!
# 
# hth

# In[ ]:




