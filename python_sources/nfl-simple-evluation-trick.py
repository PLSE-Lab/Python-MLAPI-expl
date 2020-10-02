#!/usr/bin/env python
# coding: utf-8

# # Introduction
# I will introduce a simple trick of evaluation.
# 
# This competition is scored by Continuous Ranked Probability Score (CRPS).
# The CRPS is computed as follows:
# $$
# C=\frac{1}{199N}\sum_{m=1}^N\sum_{n=-99}^{99}(P(y\geq n)-H(n-Y_m))^2
# $$
# $H(x)=1$ if $x\geq 0$ else $0$
# 
# 
# Let's consider the following to minimize this function.
# 
# **In the following, we will consider how to output when the correct answer is 0 yards and the predicted value is 5 yards.**
# 
# # case0:not predict
# 
# At first, try without using even the predicted value.<br>
# Consider the following horizontal line.

# In[ ]:


import os
import pandas as pd
from kaggle.competitions import nflrush
import random
import gc
import pickle
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


# In[ ]:


plt.plot([i-99 for i in range(199)],[1 if i-99>=0 else 0 for i in range(199)],label="ans")
plt.plot([i-99 for i in range(199)],[0.5 for i in range(199)],label="pred")
plt.legend()


# In[ ]:


print("case0-1's score is ",sum([((1 if i-99>=0 else 0)-0.5)**2 for i in range(199)])/199)


# CRPS is 0.25. This is very bad.<br>
# Next, consider the following diagonal line.

# In[ ]:


plt.plot([i-99 for i in range(199)],[1 if i-99>=0 else 0 for i in range(199)],label="ans")
plt.plot([i-99 for i in range(199)],[i/199 for i in range(199)],label="pred")
plt.legend()


# In[ ]:


print("case0-2's score is ",sum([((1 if i-99>=0 else 0)-(i/199))**2 for i in range(199)])/199)


# CRPS become 0.083! <br>
# From this example, it can be seen that CRPS represents the area surrounded by two lines.
# 
# **So what we want to do is minimize this area.**

# # case1:One step staircase
# From here, we use the predicted value, 5 feet.
# Try the same output as function $H$.
# 
# 
# The submission is [$p_{-99},p_{-98},p_{-97},p_{-96},...,p_{98},p_{99}$]
# 
# 
# $p_i = 0.0$ if $i\geq 5$ else $1.0$  

# In[ ]:


plt.plot([i-99 for i in range(199)],[1 if i-99>=0 else 0 for i in range(199)],label="ans")
plt.plot([i-99 for i in range(199)],[1 if i-99>=5 else 0 for i in range(199)],label="pred")
plt.legend()


# In[ ]:


print("case1's score is ",sum([((1 if i-99>=0 else 0)-(1 if i-99>=5 else 0))**2 for i in range(199)])/199)


# the score become 0.025. <br>
# Compared to the previous example, it is much smaller.
# 
# # case2:Line with large slope
# 
# How about making the slope of the diagonal line steep?
# 
# The submission is [$p_{-99},p_{-98},p_{-97},p_{-96},...,p_{98},p_{99}$]
# 
# $p_i = 1.0$ if $i\geq 5+W$<br>
# $p_i = \frac{i-(5-W)}{2W}$ else if $i\geq 5-W$<br>
# $p_i = 0.0$ otherwise
# 
# $W$ is window width. 

# In[ ]:


plt.plot([i-99 for i in range(199)],[1 if i-99>=0 else 0 for i in range(199)],label="ans")
plt.plot([i-99 for i in range(199)],[1 if i-99>=5+10 else 0 if i-99<5-10 else ((i-99)-(5-10))/20 for i in range(199)],label="pred")
plt.legend()


# In[ ]:


print("case2's score is ",sum([((1 if i-99>=0 else 0)
                                - (1 if i-99>=5+10 else 0 if i-99<5-10 else ((i-99)-(5-10))/20))**2 
                               for i in range(199)])/199)


# Compared to the previous 0.025, it is about 0.01 smaller!
# 
# # case3:Cumulative sum of normal distribution
# 
# So far we've only tried straight lines, but let's try other things.<br>
# Since the output is considered to follow a normal distribution,
# we can expect CRPS to be even smaller for the entire data.
# 
# The normal distribution is as follows.

# In[ ]:


from scipy.stats import norm
x = np.arange(-10,10,0.01)
y = norm.pdf(x,0,3)
plt.plot(x,y)
plt.xlim(-10,10)


# Since we want a cumulative sum, take the cumulative sum of the normal distribution values.

# In[ ]:


x = np.arange(-10,10,0.01)
y = norm.pdf(x,0,3)
plt.plot(x,np.cumsum(y))
plt.xlim(-10,10)


# It looks good.
# 
# The submission is [$p_{-99},p_{-98},p_{-97},p_{-96},...,p_{98},p_{99}$]
# 
# $p_i = 1.0$ if $i\geq 5+W$<br>
# $p_i = P(\mathcal{N}(0,3) \leq i-5)$ else if $i\geq 5-W$<br>
# $p_i = 0.0$ otherwise
# 
# $W$ is window width. 

# In[ ]:


norm_cumsum = np.cumsum(norm.pdf(np.arange(-10,10,1),0,3))

plt.plot([i-99 for i in range(199)],[1 if i-99>=0 else 0 for i in range(199)],label="ans")
plt.plot([i-99 for i in range(199)],[1 if i-99>=5+10 else 0 if i-99<5-10 else norm_cumsum[(i-99)-(5-10)] for i in range(199)],label="pred")
plt.legend()


# In[ ]:


print("case3's score is ",sum([((1 if i-99>=0 else 0)
                                - (1 if i-99>=5+10 else 0 if i-99<5-10 else 
                                   norm_cumsum[(i-99)-(5-10)]))**2 
                               for i in range(199)])/199)


# In this case, the score was inferior to case2.
# 
# # evaluation
# 
# I will try to find out how much score is easily obtained from the training data.

# In[ ]:


env = nflrush.make_env()
train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)


# In[ ]:


print("the median value of train data is ",np.median(train_df["Yards"]))


# In[ ]:


y_train = np.array([train_df["Yards"][i] for i in range(0,509762,22)])
y_pred_case1 = np.zeros((509762//22,199))
y_pred_case2 = np.zeros((509762//22,199))
y_pred_case3 = np.zeros((509762//22,199))
y_ans = np.zeros((509762//22,199))
norm_cumsum = np.cumsum(norm.pdf(np.arange(-10,10,1),0,3))

p=3
w=10
for i in range(509762//22):
    for j in range(199):
        if j>=p+w:
            y_pred_case2[i][j]=1.0
            y_pred_case3[i][j]=1.0
        elif j>=p-w:
            y_pred_case2[i][j]=(j+w-p)/(2*w)
            y_pred_case3[i][j]=norm_cumsum[max(min(j+w-p,19),0)]
        if j>=p:
            y_pred_case1[i][j]=1.0

for i,p in enumerate(y_train):
    for j in range(199):
        if j>=p:
            y_ans[i][j]=1.0

print("validation score in case1:",np.sum(np.power(y_pred_case1-y_ans,2))/(199*(509762//22)))
print("validation score in case2:",np.sum(np.power(y_pred_case2-y_ans,2))/(199*(509762//22)))
print("validation score in case3:",np.sum(np.power(y_pred_case3-y_ans,2))/(199*(509762//22)))


# case2 and case3 are much better than simple case1! And,in the previous example, case3 was defeated by case2, but in actual data, case3 is slightly better than case2!
# 
# I think that it is useful as a process after applying the regression algorithm.
# 
# Please let me know if you have any opinions or advice.
