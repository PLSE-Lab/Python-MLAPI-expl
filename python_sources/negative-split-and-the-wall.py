#!/usr/bin/env python
# coding: utf-8

# In this script I will explore 2 ideas with which many runners are highly familiar:
# 
#  1. [The negative split][1] strategy, in which the runner runs the 2nd part faster than the 1st part. A running strategy often recommended by some coaches and running enthusiasts
#  2. The phenomenon of ["The Wall"][2] - a condition of sudden fatigue which typically hits the marathon runner after about 30Ks (though of course varies among different individuals)
# 
# Can we detect these in our data?  
#  
# 
# 
#   [1]: http://running.competitor.com/2016/06/training/why-negative-splits-are-ideal-on-race-day_152209
#   [2]: https://en.wikipedia.org/wiki/Hitting_the_wall

# Load Libraries 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
plt.style.use('fivethirtyeight')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Read The Data

# In[ ]:


df = pd.read_csv('../input/marathon_results_2016.csv')


# Just an auxiliary function and a test

# In[ ]:


def time_to_min(string):
    if string is not '-':
        time_segments = string.split(':')
        hours = int(time_segments[0])
        mins = int(time_segments[1])
        sec = int(time_segments[2])
        time = hours*60 + mins + np.true_divide(sec,60)
        return time
    else:
        return -1

print(time_to_min(df.loc[1,'Half']))


# ## The Negative split
# 
# Is this really a common strategy? let us plot the split ratio distribution. That is, the time ratio between the first and a second half of the race. A value smaller than 1 implies a negative split - i.e, the 2nd part of the race was faster than the first one. And inversely, a value which is larger than 1 indicates the more intuitive (and evidently more common) positive split:

# In[ ]:


df['Half_min'] = df.Half.apply(lambda x: time_to_min(x))
df['Full_min'] = df['Official Time'].apply(lambda x: time_to_min(x))
df['split_ratio'] = (df['Full_min'] - df['Half_min'])/(df['Half_min'])
df_split = df[df.Half_min > 0]
sns.kdeplot(df_split.split_ratio)
plt.xlim([0.7,1.7])
plt.xlabel('Split Ratio')
plt.title('Split Distribution (Negative split when < 1)')


# As we can see, most of the runners chose (or were compelled to choose, limited by the constraints their body imposes upon them) a positive split - They run the 2nd half slower than the first half. for most of them, however, the difference is small and they are able to maintain a relatively constant pace. There's a long tail on the right of runners who slowed down significantly. On the left we can see a small minority of runners who did in fact run a negative split.
# 
# Let's see whether this correlates with the general performance:

# In[ ]:


plt.plot(df_split.Overall,df_split.split_ratio,'o', alpha = 0.2)
plt.ylim([0.5,3])
plt.xlabel('Overall Rank')
plt.ylabel('Split')
plt.title('Split and performance')


# We can see that while the general trend does show a very moderate increase in the split ratio with the overall rank (that is - the lower the runner is ranked, the more positive his or her split is). However, the most dominant trend is the growing variance: while the good runners are all concentrated around a split of, say, 1.2, the worse runners are all around the place.

# Let's see whether the split strategy or performance depends on any demographic features.
# 
# First, males vs females:

# In[ ]:


sns.distplot(df_split.split_ratio[df_split['M/F'] == 'M'], np.arange(0.6,3,0.01))
sns.distplot(df_split.split_ratio[df_split['M/F'] == 'F'],np.arange(0.6,3,0.01))
plt.legend(['Males','Females'])
plt.xlim([0.5,2])


# The Average is very close, but the longer tail of the males distribution skews it to the right.
# 
# This can of course be a result of another difference between the populations that affects the split ratio distribution. 

# But first, let's see how the split changes with age:

# In[ ]:


median = np.median(df_split.Age)
sns.distplot(df_split.split_ratio[df_split.Age > median],np.arange(0.6,3,0.01))
sns.distplot(df_split.split_ratio[df_split.Age < median],np.arange(0.6,3,0.01))
plt.xlim([0.5,2])
plt.legend(['Age >' + str(median), 'Age <' + str(median)])


# The two distributions virtually overlap. Let's live the median runner and look at the very young and very old: one can expect the stamina will strongly depend on age (though this effect can of course be cancelled when experience kicks in):

# In[ ]:


sns.distplot(df_split.split_ratio[df_split.Age >50],np.arange(0.6,3,0.01))
sns.distplot(df_split.split_ratio[df_split.Age <30],np.arange(0.6,3,0.01))
plt.xlim([0.5,2])
plt.legend(['Age > 50', 'Age < 30'])


# No, we can quite conclusively say that the split rate is not age dependent.
# 
# Let's see now the general performance correlates with the splits:

# In[ ]:


sns.distplot(df_split.split_ratio[df_split.Overall < 1000],np.arange(0.6,3,0.01))
sns.distplot(df_split.split_ratio[df_split.Overall > 10000],np.arange(0.6,3,0.01))
plt.legend(['Overall <1000','Overall>10000'])
plt.xlim([0.5,2])


# The better runners have a more negative (or less positive, to be more accurate) than worse runners.
# 
# This is somewhat trivial - a negative split is limited: unless a runner deliberately runs far slower than he can in the first half, it is very unlikely for a runner to run the second part significantly faster **and** finish in a good place. The opposite scenario, however, is much more likely: a runner starts too fast, and then slows down as he or she gets tired. As there is no upper limit to the level of collapse, the split can be very positive, and the more positive it is the worse the rank can be.

# Runners from Ethiopia and Kenya are over represented in the list of the top 100 runners. Is their split different than the other runners who finished top 100 (in that way we partially control for general performance. partially, because the runners from these countries dominate the top 20 places as well):

# In[ ]:


plt.plot(df_split.Overall[(df_split.Country is not 'ETH') & (df_split.Country is not 'KEN') & (df_split.Overall<100)], df_split.split_ratio[(df_split.Country is not 'ETH') & (df_split.Country is not 'KEN') & (df_split.Overall<100)],'o')
plt.plot(df_split.Overall[(df_split.Country == 'ETH') & (df_split.Overall<100)], df_split.split_ratio[(df_split.Country == 'ETH')  & (df_split.Overall<100)],'o', color = 'r')
plt.plot(df_split.Overall[(df_split.Country == 'KEN') & (df_split.Overall<100)], df_split.split_ratio[(df_split.Country == 'KEN')  & (df_split.Overall<100)],'o', color = 'r')
plt.xlabel('Overall Rank')
plt.ylabel('Split Ratio')
plt.legend(['Others','Kenya and Ethiopia'])


# This sample size is of course way too small for a robust statistical conclusion, but it does seem as if runners from Ethiopia and Kenya tend to have more negative splits, even when controlling for general performance. 
# 
# out of 5 runners with negative splits among the top 40, 4 were from these countries even though there were only 13 from these countries in this list. 

# ## Race stability
# 
# Do better runners keep a more constant pace along the race? (Probably - but how significant is the effect?)

# In[ ]:


df['5K_mins'] = df['5K'].apply(lambda x: time_to_min(x))
df['10K_mins'] = df['10K'].apply(lambda x: time_to_min(x))
df['10K_mins'] = df['10K_mins'] - df['5K_mins'] 
df['15K_mins'] = df['15K'].apply(lambda x: time_to_min(x))
df['15K_mins'] = df['15K_mins'] - df['10K_mins'] -  df['5K_mins']
df['20K_mins'] = df['20K'].apply(lambda x: time_to_min(x))
df['20K_mins'] = df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] -  df['5K_mins']
df['25K_mins'] = df['25K'].apply(lambda x: time_to_min(x))
df['25K_mins'] = df['25K_mins'] - df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] -  df['5K_mins']
df['30K_mins'] = df['30K'].apply(lambda x: time_to_min(x))
df['30K_mins'] = df['30K_mins'] -df['25K_mins'] - df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] -  df['5K_mins']
df['35K_mins'] = df['35K'].apply(lambda x: time_to_min(x))
df['35K_mins'] = df['35K_mins'] -df['30K_mins'] -df['25K_mins'] - df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] -  df['5K_mins']
df['40K_mins'] = df['40K'].apply(lambda x: time_to_min(x))
df['40K_mins'] = df['40K_mins'] -  df['35K_mins'] -df['30K_mins'] -df['25K_mins'] - df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] -  df['5K_mins']


# In[ ]:


columns = ['40K_mins','35K_mins','30K_mins','25K_mins','20K_mins','15K_mins','10K_mins','5K_mins']
df['avg'] = df[columns].mean(axis = 1)
df['stdev'] = df[columns].std(axis = 1)
df_split = df[(~(df['5K'] == '-')) &(~(df['10K'] == '-'))&(~(df['15K'] == '-'))&(~(df['20K'] == '-'))&(~(df['25K'] == '-')) &(~(df['30K'] == '-')) &(~(df['35K'] == '-')) &(~(df['40K'] == '-'))]


# In[ ]:


plt.plot(df_split.avg,df_split.stdev,'o', alpha = 0.1)
plt.ylim([0,30])
plt.xlim(15,45)
plt.xlabel('Average Time for 5K along the race')
plt.ylabel('Standard Deviation of segments pace')
plt.title('Faster runners are also more stable')


# Unsurprisingly, we can see that the faster the runner is, the more stable, on average, he is.

# ## The Wall
# 
# Can we find it? Is there a distance where many runners drastically slow down?
# 
# First let's add features that represent the slowing down in each segment compared to its previous:

# In[ ]:


df_split['10_dif'] = df_split['10K_mins'] - df_split['5K_mins'] 
df_split['15_dif'] = df_split['15K_mins'] - df_split['10K_mins']
df_split['20_dif'] = df_split['20K_mins'] - df_split['15K_mins']
df_split['25_dif'] = df_split['25K_mins'] - df_split['20K_mins']
df_split['30_dif'] = df_split['30K_mins'] - df_split['25K_mins']
df_split['35_dif'] = df_split['35K_mins'] - df_split['30K_mins']
df_split['40_dif'] = df_split['40K_mins'] - df_split['25K_mins']


# Now let's see whether the slowing down distribution for each segment moves with distance, and especially, whether we can detect a specific distance where the histogram shifts much more:

# In[ ]:


sns.distplot(df_split['10_dif'],np.arange(-5,10,0.08), kde = False)
sns.distplot(df_split['20_dif'],np.arange(-5,10,0.08), kde = False)
sns.distplot(df_split['30_dif'],np.arange(-5,10,0.08), kde = False)
sns.distplot(df_split['40_dif'],np.arange(-5,10,0.08), kde = False)
plt.legend(['10K','20K','30K','40K'])
plt.xlabel('Slowing Down Compared to previous segment')


# We can see that the 10K segment (from 5K to 10K) is usually as fast as it's previous. similarly, the 15-20K is slightly slower, with the histogram keeps being pretty narrow. the 25-30K and and the 35-40K ones show a much more obvious slowing down.
# 
# Can this be regarded as the wall, or is the widening of the histogram, with more and more runners slowing down, should be regarded as a simple continuous process? Let's zoom in on both halves of the race:

# In[ ]:


sns.distplot(df_split['25_dif'],np.arange(-5,10,0.05), kde = False)
sns.distplot(df_split['30_dif'],np.arange(-5,10,0.05), kde = False)
sns.distplot(df_split['35_dif'],np.arange(-5,10,0.05), kde = False)
sns.distplot(df_split['40_dif'],np.arange(-5,10,0.05), kde = False)
plt.xlim([-5,10])
plt.legend(['25K','30K','35K','40K'])
plt.xlabel('Slowing Down Compared to previous segment')


# Interestingly, the 30-35K segment has a similar **slowing rate** as it's previous segment (it is still slower of course!), having the next increase in **slow down rate** at 35-40. this does imply that the 25-30 segment is more wall-ish, but the last segment is actually the worse.
# 
# We can probably say that the slowing down rate is not constant nor linear, and there are some "melting points" which are different for different runners. 
# 
# Let's see how does the slowing down look in the first half of the race:

# In[ ]:


sns.distplot(df_split['10_dif'],np.arange(-5,10,0.05), kde = False)
sns.distplot(df_split['15_dif'],np.arange(-5,10,0.05), kde = False)
sns.distplot(df_split['20_dif'],np.arange(-5,10,0.05), kde = False)
sns.distplot(df_split['25_dif'],np.arange(-5,10,0.05), kde = False)
plt.xlim([-5,10])
plt.legend(['10K','15K','20K','25K'])
plt.xlabel('Slowing Down Compared to previous segment')


# The behavior is clearly very different (and similar to the 20-25K segment which can be seen in the previous plot).
# 
# Therefore we can indeed say that a series of walls, hitting different runners in different distances, can be found after the 25 kilometer.
