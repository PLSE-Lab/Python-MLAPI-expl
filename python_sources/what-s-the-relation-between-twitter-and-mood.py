#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import json
import pylab as plt
from scipy.stats.mstats import zscore


# # Hypothesis: there is a relationship between Twitter usage and self-reported mood
# ## Reading in cleaned data
# Also dropping rows with missing values.

# In[ ]:


combined_df = pd.read_csv('../input/data-cleaning/combined_measures.csv', index_col='date').dropna()


# # Exploration
# Exploring relationships between self-reported mood and two Twitter usage measures: number of tweets posted and number of tweets I was mentioned in.

# In[ ]:


sns.pairplot(combined_df[['tweets', 'twitter_mentions', 'mood_baseline', 'mood_cat']], hue="mood_cat")


# Unexpectadly the two measures of twitter usage are correlated.

# In[ ]:


from scipy import stats
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2
sns.jointplot(y="tweets", x="twitter_mentions", data=combined_df, kind='reg', stat_func=r2)


# The two variables share 48% of the variance
# ## Can **number of tweets** explain differences in self reported mood?
# Plotting the relationships

# In[ ]:


sns.boxplot(y="tweets", x="mood_cat", data=combined_df,
            whis="range")


# Formal hypothesis testing using a linear model

# In[ ]:


X = combined_df[['tweets', 'mood_baseline']]
X = sm.add_constant(X)
y = combined_df['mood_int']

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()


# In[ ]:


sns.regplot(y="tweets", x="mood_int", data=combined_df, x_estimator=np.mean)


# Conclusion: Number of posted tweets does not explain mood.
# ## Can **number of twitter mentions** explain differences in self reported mood?
# Plotting the relationships

# In[ ]:


sns.boxplot(y="twitter_mentions", x="mood_cat", data=combined_df,
            whis="range")


# Formal hypothesis testing

# In[ ]:


X = combined_df[['twitter_mentions', 'mood_baseline']]
X = sm.add_constant(X)
y = combined_df['mood_int']

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()


# In[ ]:


sns.regplot(y="twitter_mentions", x="mood_int", data=combined_df, x_estimator=np.mean)


# Number of twitter mentions also does not explain mood.
# ## Combined relationship betwee the number of tweets and mentions and the mood
# How much mood variance can be explained by a linera combination of twitter mentions and tweets?

# In[ ]:


X = combined_df[['twitter_mentions', 'tweets', 'mood_baseline']]
X = sm.add_constant(X)
y = combined_df['mood_int']

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()


# Even the combination of the two do not explain mood.
# ## Caveats 
# * There are many "unseen" variables that can influence both mood and Twitter behaviour - for example mood can be better when hiking which is when less twitting also happens
# * This is only a correlational study. An intervention (stop tweeting on random days) would have to be performed to check for causality (we don't know if bad mood causes more tweeting or more tweeting causes bad mood)****

# In[ ]:




