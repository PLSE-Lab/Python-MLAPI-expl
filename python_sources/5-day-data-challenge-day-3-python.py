#!/usr/bin/env python
# coding: utf-8

# ## 5 Day Data Challenge:
# The Kernel contains a small tutorial for those who want to walkthrough the Data Science procedure of investigating and exploring a dataset. The source of this kernel stems from Rachael Tatman the-5-day-data-challenge. The language that will be used is python.

# ### Day 3: Perform a t-test

# Today we're going to do a t-test, but first let's read in our data & set up our environment.

# In[ ]:


# load in our libraries
import pandas as pd # pandas for data frames
from scipy.stats import ttest_ind # just the t-test from scipy.stats
from scipy.stats import probplot # for a qqplot
import matplotlib.pyplot as plt # for a qqplot
import pylab #

# read in our data
cereals = pd.read_csv("../input/cereal.csv")
# check out the first few lines
cereals.head()


# We should make sure that the variable is normally distributed (spoiler: I checked that it was beforehand ;) so let's use a qq-polt to do that.

# In[ ]:


# plot a qqplot to check normality. If the varaible is normally distributed, most of the points 
# should be along the center diagonal.
probplot(cereals["sodium"], dist="norm", plot=pylab)


# Preform our t-test.

# In[ ]:


# get the sodium for hot cerals
hotCereals = cereals["sodium"][cereals["type"] == "H"]
# get the sodium for cold ceareals
coldCereals = cereals["sodium"][cereals["type"] == "C"]

# compare them
ttest_ind(hotCereals, coldCereals, equal_var=False)


# So, if we decided to set our alpha (the highest p-value that would still allow us to reject the null) to 0.05 before we started, we would reject the null (i.e. can be pretty sure that there's not not a difference between these two groups). Statistic is the actual value of the t-test, and the pvalue is the probability that we saw a difference this large between our two groups just due to chance if they were actually drawn from the same underlying population.

# In[ ]:


# let's look at the means (averages) of each group to see which is larger
print("Mean sodium for the hot cereals:")
print(hotCereals.mean())

print("Mean sodium for the cold cereals:")
print(coldCereals.mean())


# Now plot for the two cereal types, with each as a different color.

# In[ ]:


# plot the cold cereals
plt.hist(coldCereals, alpha=0.5, label='cold')
# and the hot cereals
plt.hist(hotCereals, label='hot')
# and add a legend
plt.legend(loc='upper right')
# add a title
plt.title("Sodium(mg) content of cereals by type")

