#!/usr/bin/env python
# coding: utf-8

# # Cleaning data with Python - Challenge Day 2 - Scaling and normalising data
# 
# Back in my days of working with gene expression microarray data, scaling and normalising was always part of the workflow, but I didn't do it in Python. Time to learn how with [Day 2 of Rachael's challenge][1]...
# [1]: https://www.kaggle.com/rtatman/data-cleaning-challenge-scale-and-normalize-data
# 

# In[ ]:


# modules we'll use
import pandas as pd
import numpy as np

# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# read in all our data
kickstart = pd.read_csv("../input/ks-projects-201801.csv")

# set seed for reproducibility
np.random.seed(0)


# ### For the following examples, decide whether scaling or normalization makes more sense.
# 
# 1. You want to build a linear regression model to predict someone's grades given how much time they spend on various activities during a normal school week. You notice that your measurements for how much time students spend studying aren't normally distributed: some students spend almost no time studying and others study for four or more hours every day. Should you scale or normalize this variable?
# 
# *This one's a bit of a tricky one, from the question, we've already decided we want to build a linear regression model before we've really gotten stuck into looking at the data. Is a linear regression model the most appropriate? Are we thinking of transforming the dataset prior to regression so we **can** use a linear model, which might not fit untransformed data?
# 
# Additionally, my understanding of linear regression is that the assumption of normality is more important for the residual errors, rather than the variables and that, with a sufficient sample size, even that isn't critical.
# 
# However, as I'm guessing all of this has already been thought of as it is a practice question, we might want to normalise the dataset prior to building our regression model.
# 
# 2. You're still working on your grades study, but you want to include information on how students perform on several fitness tests as well. You have information on how many jumping jacks and push-ups each student can complete in a minute. However, you notice that students perform far more jumping jacks than push-ups: the average for the former is 40, and for the latter only 10. Should you scale or normalize these variables?
# 
# *I would scale in this case: scaling would allow us to more easily compare the performance across different exercises*
# 
# ### Your turn! We looked as the usd_pledged_real column. What about the "pledged" column? Does it have the same info?

# In[ ]:


indexOfPositivePledged = kickstart.pledged > 0

# get only positive pledges (using their indexes)
positivePledged = kickstart.pledged.loc[indexOfPositivePledged]

# normalise the pledges (w/ Box-Cox)
normalisedPledged = stats.boxcox(positivePledged)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positivePledged, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalisedPledged, ax=ax[1])
ax[1].set_title("Normalised Data")


# That concludes the first part of the challenge. However, there is an extra 'More Practice' exercise:
# 
# Try finding a new dataset and pretend you're preparing to preform a regression analysis. Pick three or four variables and decide if you need to normalize or scale any of them and, if you think you should, practice applying the correct technique.
# 
# But, for that, [I might switch to R...][1]
# [1]: https://www.kaggle.com/chrisbow/cleaning-data-with-python-day-2-more-practice
