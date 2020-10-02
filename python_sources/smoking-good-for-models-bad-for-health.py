#!/usr/bin/env python
# coding: utf-8

# <img src="https://i.imgur.com/qlslweP.png" width="600">
# 
# <br>
# Cigarette smoking is thought to have a strong correlation with pulmonary fibrosis. I wanted to check that out through the lens of Survival Analysis.

# In[ ]:


get_ipython().system(' conda install -c conda-forge lifelines -y')


# In[ ]:


import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt


# The first thing here is to get a fairly homogenous group with respect to other correlated factors, namely gender and age. I chose to focus on males over 60. The first thing you see is there are a lot of smokers and former smokers in this group. A study I saw estimated that ~10% of US citizens either smoke or have smoked 100+ cigarettes in their lifetime. The proportion here is more than reversed!

# In[ ]:


train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
train = train.query('Sex=="Male" & Age>60')
train.SmokingStatus.value_counts(normalize=True)


# To get more detail on the effect of smoking, I used Survival Analysis. Survival analysis was developed to model the expected time until an event happens for members of a given population. It's a great way to look at data like this. In this case, I applied it to estimate how long patients might go without losing a certain amount of lung capacity, and to see if smoking affects the probability of losing lung capacity.
# 
# In looking at survival rates, I defined "non-survival" as having a lung capacity (FVC) below the average lung capacity for all men over 60. The chart below shows the mean value and distribution.

# In[ ]:


display(train.FVC.hist(),
train.FVC.mean())


# It's a good practice to look at the division of patients and make sure there are adequate sample sizes for both groups.

# In[ ]:


train['LowFVC'] = train.FVC.lt(train.FVC.mean()).astype(int)
train.LowFVC.value_counts()


# The chart below shows the survival rates. For each group you can see the proportion of players who drop out over time, where time is defined as the number of weeks since the patient's baseline CT scan. The size of the group with above average lung capacity decreases from left to right as the disease progresses. 
# 
# The solid lines represent the observed survival rate for each group. When one curve is below the other it represents a lower survival rate; i.e., a higher rate of patients whose lung capacity drops below the mean.
# 

# In[ ]:


CONFIDENCE = 0.90

idx1 =  (train.SmokingStatus == 'Never smoked')
idx2 = (train.SmokingStatus.isin(['Ex-smoker', 'Currently smokes']))

durations1 = train.loc[idx1, 'Weeks']
durations2 = train.loc[idx2, 'Weeks']

events1 = train.loc[idx1, 'LowFVC']
events2 = train.loc[idx2, 'LowFVC']

kmf1 = KaplanMeierFitter()
kmf1.fit(durations1, events1, alpha=(1-CONFIDENCE), label='Never Smoked')

kmf2 = KaplanMeierFitter()
kmf2.fit(durations2, events2, alpha=(1-CONFIDENCE), label='Smoked')

plt.clf()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(12,8))
plt.style.use('seaborn-whitegrid')
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)


p1 = kmf1.plot()
p2 = kmf2.plot(ax=p1)


plt.xlim(-5, 118)
plt.title("Maintaining Lung Capacity")
plt.xlabel("Weeks since baseline CT")
plt.ylabel("Fraction of Group with above average FVC")
plt.show()


# The shaded areas represent the 90% confidence interval. Confidence intervals that do not overlap indicate a meaningful difference between the groups. In this case the confidence intervals overlap until around 40 weeks, when it appears that the intervals diverge.
# 
# An example takeaway: At 60 weeks - just over a year - it looks like about 35% of the non-smokers still have above-average lung capacity. The smokers are at around 25%. That's only 2/3 the fraction of non-smokers, which seems like quite a difference.
# 
# As suspected, smoking status is relevant and good for models. But it's not good for health, so if you smoke, please consider limiting it!
