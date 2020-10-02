#!/usr/bin/env python
# coding: utf-8

# # Ramen Lovers Love Eating...
# 
# What makes good ramen, good ramen? At least in the United States, basis of ramen-loving has, at least for the last few decades, come out of the starving poor college student demographic. For lots of undergraduates (and post-graduates too), college expenses and rents preclude eating on anything more than shoe-string budget. If you're a broke-ass college student, ramen probably practical runs in your veins, you eat it so often. Cause it's so cheap. And if you fall in love with it so hard that you continue looking out for those rare ramens years after college, then you're a ramenphile.
# 
# OK, but about something fancier, like wine? Unless you grew up learning vineculture from Papa Pierre on a French villa, wine is an expensive luxury, positioned as the alcohol-cum-excellence for any affair worth throwing. So naturally the wine-loving community will be of the snoottier variety. Where I live, in New York, financiers are constantly "discovering" a deeply-held passion for viticulture not long after acquiring their bulge-bracket bank jobs.
# 
# I propose that ramen lovers are much more easily satisfied than wine lovers. To test this theory, let's look at the ratings given out by The Ramen Rater, and compare them to those handed out by Wine Magazine.

# In[ ]:


import pandas as pd
ramen = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv")
wine = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('fivethirtyeight')

sns.countplot(np.round(ramen['Stars'].replace('Unrated', np.nan).dropna().astype(np.float64)))
plt.suptitle("Ramen Ratings")


# In[ ]:


plt.suptitle("Wine Ratings")
sns.countplot(np.round((wine['points'].dropna() - 80) / 4))


# So wine lovers think most wines are crap, ramen lovers think more ramens are pretty good.
# 
# And there it is! Class differences in a nutshell.

# ## Addendum
# 
# The overall shape is the same if we increase to 10 categories (as you would expect), but this is a bit more accurate/informative:

# In[ ]:


sns.countplot(
    np.round(ramen['Stars'].replace('Unrated', np.nan).dropna().astype(np.float64) * 2) / 2
)
plt.suptitle("Ramen Ratings")


# In[ ]:


sns.countplot(np.round((wine['points'].dropna() - 80) / 2) / 2)

