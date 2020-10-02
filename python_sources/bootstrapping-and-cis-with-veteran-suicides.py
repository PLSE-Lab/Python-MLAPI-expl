#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This very sobering dataset documents the occurances of suicides between veterans and non-veterans by state. As I stated in the introduction, state-side suicide rates amongst veterans are highly elevated over those of ordinary civilians. We can use this dataset to understand *how much* higher they are by building a confidence interval (often shorthanded to CI) around the data.
# 
# ## Confidence Intervals
# 
# You may also want to read the [hypothesis testing](https://www.kaggle.com/residentmario/hypothesis-testing-with-firearm-licensees/) notebook. The salient point from that notebook was the fact that, due to the Central Limit Theorem, an estimator drawn more than approximately 30 times will have errors (deviations from the true value) that are approximately normally distributed.
# 
# To build a confidence interval, we first center on our estimator: in this case, the mean of the statewide suicide rates is an estimate for the nationwide rate, $\bar{x}$. We then plug it into the following formula:
# 
# $$\text{CI} = \left(\bar{x} - t^*\frac{s}{\sqrt{n}}, \bar{x} + t^*\frac{s}{\sqrt{n}}\right)$$
# 
# Where $s$ is the [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation) of the dataset (the square root of its variance, a population parameter describing the amount of spread in the data); $t^*$ is the t-score; and $n$ is the number of samples.
# 
# A standard normal distribution is centered on the mean $\bar{x}$, with decreasing probability as we go further out to the left or right of the distribution. $t^*$ describes how far to the left and how far out on a standard normal distribution we are going to try to capture "enough" of the distribution to be meaningfully accurate. The more confidence we want to be in our confidence interval, the larger the interval will be, because the more cumulative probability we wil have to cover. This is an accuracy-precision tradeoff.
# 
# $t^*$ is based on the amount of confidence we want to have (95%, 99%, etc.) and also on the number of sampes in the distribution. A distribution with a very low number of samples will have significantly longer intervals. In other words, the errors will be distributed according to the standard normal only in the case when the number of samples is large enough! In cases with very few points, a version of the normal with longer tails, the [Student's t-distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution), will be used instead. This limiting case, the standard normal, is what we can stick to in most practical cases ($n \geq 15$ or so) however; it is also known as the $z$ score.
# 
# ## Implementation
# 
# With that jargon out of the way let's go ahead and write up a confidence interval!

# In[4]:


import pandas as pd
import numpy as np
veterans_2005 = pd.read_csv("../input/2005.csv", index_col=0)
veterans_2005.head(3)


# In[13]:


df_2005 = pd.DataFrame(
    {'vet': veterans_2005['vet_suicides'] / veterans_2005['vet_pop'],
     'civ': (veterans_2005['all_suicides'] - veterans_2005['vet_suicides']) / 
            (veterans_2005['overall_pop_18'] - veterans_2005['vet_pop'])}
)
df_2005.head(3)


# The formula is relatively straightforward to implement, though finding the $t$ slash $z$ score requires a table lookup in `scipy`.

# In[40]:


import scipy.stats as st

def confidence_interval(X, c):
    x_bar = np.mean(X)
    z_score = st.norm.ppf(1 - ((1 - c) / 2))
    sqrt_n = np.sqrt(len(X))
    std_dev = np.std(X)
    
    delta = z_score * (std_dev / sqrt_n)
    return np.array([x_bar - delta, x_bar + delta])


# Here are our 95% confidence intervals for suicide rates in 2005 for civilian and veteran populations, respectively.

# In[41]:


confidence_interval(df_2005.civ, 0.95)


# In[42]:


confidence_interval(df_2005.vet, 0.95)


# You can clearly see a *very* large difference in these confidence intervals. However, the numbers themselves are hard to interpret, so let's multiply things through a bit. The following values are 95% confidence intervals for the annualized suicide risk for civilians and veterans, respectively, per one million population.

# In[43]:


confidence_interval(df_2005.civ, 0.95) * 1000000


# In[44]:


confidence_interval(df_2005.vet, 0.95) * 1000000


# Non-military civilian suffer approximately 150 suicides per million per year. We are 95% confident the true mean lies between 142 and 166. Veterans, meanwhile, suffer approximately 300 suicides per year. We are 95% confident the true mean lies between 274 and 331.
# 
# Overall, the data backs up the fact that **veterans commit suicide twice as often as non-military civilians do** (this result is corraborated by research summarized e.g. [here](https://en.wikipedia.org/wiki/United_States_military_veteran_suicide)).

# ## Bootstrapping
# 
# I'm going to now introduce another concept, bootstrapping, which will help illustrate why confidence intervals work the way they do.
# 
# Bootstrapping is an extremely useful technique for non-parameterically (that is, without paying attention to distributions) estimating things that we are interested in estimating. We bootstrap data by randomly sampling estimates from our data and applying our estimator to the random samples. The samples should be a small but significant slice of the overall data.
# 
# In this illustrative case I'll use $n=20$.

# In[61]:


draws = np.array([np.random.choice(df_2005.civ, size=20) for _ in range(10000)]) * 1000000
civ_means = np.array([np.mean(draw) for draw in draws])

draws = np.array([np.random.choice(df_2005.vet, size=20) for _ in range(10000)]) * 1000000
vet_means = np.array([np.mean(draw) for draw in draws])

del draws


# In[82]:


import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

fig, axarr = plt.subplots(1, 2, figsize=(12, 4))
plt.suptitle("Suicides Per Million Estimators, Bootstrapped").set_position([.5, 1.05])

import seaborn as sns
sns.distplot(pd.Series(civ_means), ax=axarr[0])
axarr[0].set_title("Civilians")

sns.distplot(pd.Series(vet_means), ax=axarr[1])
axarr[1].set_title("Veterans")
pass


# Bootstrapping is great visual tool because it lets us get a handle on what numbers like "we are 95% confident" mean. In this case we see that, indeed, about 95 percent of our 20-state averages are between 142 and 166, and a similar story emerges with the 274 and 331 values for veterans.

# ## Change over Time
# 
# The analysis above was based on 2005 data. What has the change in the intervening years in the data been like? Has the military made progress on or made an impact on "closing the gap" in this troubling mental health outcome?

# In[96]:


years = range(2005, 2012)

cis = []
for year in years:
    df = pd.read_csv("../input/{0}.csv".format(year), index_col=0)
    df = pd.DataFrame(
        {'vet': df['vet_suicides'] / df['vet_pop'],
         'civ': (df['all_suicides'] - df['vet_suicides']) / 
                (df['overall_pop_18'] - df['vet_pop'])}
    )
    cis.append({'civ': confidence_interval(df.civ, 0.95),
                'vet': confidence_interval(df.vet, 0.95)})


# In[107]:


civ_means = [np.mean(c['civ'])*10**6 for c in cis]
vet_means = [np.mean(c['vet'])*10**6 for c in cis]

civ_mins = [c['civ'][0]*10**6 for c in cis]
vet_mins = [c['vet'][0]*10**6 for c in cis]

civ_maxs = [c['civ'][1]*10**6 for c in cis]
vet_maxs = [c['vet'][1]*10**6 for c in cis]


# In[122]:


ind = pd.Index(range(2005, 2012))

fig, axarr = plt.subplots(1, 2, figsize=(12, 4))
plt.suptitle("Suicides Per Million Estimates, 2005-2011").set_position([.5, 1.05])

pd.Series(civ_means, index=ind).plot.line(color='black', ax=axarr[0])
pd.Series(civ_mins, index=ind).plot.line(color='steelblue', ax=axarr[0])
pd.Series(civ_maxs, index=ind).plot.line(color='steelblue', ax=axarr[0])
axarr[0].set_title("Civilians")

pd.Series(vet_means, index=ind).plot.line(color='black', ax=axarr[1])
pd.Series(vet_mins, index=ind).plot.line(color='steelblue', ax=axarr[1])
pd.Series(vet_maxs, index=ind).plot.line(color='steelblue', ax=axarr[1])
axarr[1].set_title("Veterans")
pass


# No. In fact it appears that suicides are increasing in *both* the civilian and veteran populations. =(
# 
# That's all folks. Thanks for reading!
