#!/usr/bin/env python
# coding: utf-8

# # Log-loss with New York City Building Sales
# 
# ## Discussion
# 
# Note: this notebook is an addendum to [the Classification Metrics notebook](https://www.kaggle.com/residentmario/classification-metrics-with-seattle-rain/).
# 
# Log-loss is a model performace metric which is often used within competition contexts (it is, for example, a very popular metric in Kaggle competitions). Unlike the rest of the model metrics covered in the notebook above, it involves a reasonably complicated mathematical formula.
# 
# To define log-loss mathematically, start by defining $p_{ij}$ as the probability the model will assign label $j$ to record $i$; $N$ as the number of records; $M$ as the number of class labels, and $y_{ij}$ as an indicator variable which is 1 if record $i$ is assigned class $j$ by the model, and 0 otherwise.
# 
# Then log-loss is "simply":
# 
# $$\text{logloss}(\cdot) = \frac{-1}{N}\sum_i^N \sum_j^M y_{ij} \log{p_{ij}}$$
# 
# In other words, log-loss is a logarithmic transform of the sum of the *probabilities* the model assigns to the records it misclassifies. [This excellent blog post](http://www.exegetic.biz/blog/2015/12/making-sense-logarithmic-loss/), the material inspiration for this notebook, has the following illustration of the effect this has:
# 
# ![](https://i.imgur.com/N1EzTiq.png)
# 
# Log-loss exponentially penalises misclassifications that are highly confident. In the log-loss world, a 0 observation which is predicted with 80% confidence to be a 1 observation (and classified thusly) is penalized *much* more heavily than a miscalled observation made with 55% confidence. So, in cases in which the result is wrong, it is much better for a model to be *somewhat wrong* than *extremely wrong*.
# 
# In this way it's a bit like linear regression, in that extreme outliers will give exponentially bad log-loss results. Hence the log-loss metric rewards "honest" models and rewards models that find ways to deal with outliers. This makes it an intuitively good metric to use for cases where getting reasonable classifications for *all* records, not just some or the bulk of them, is important (and this need occurs in many real-world contexts).
# 
# What is a good log-loss? A log-loss of around 0.693 is random performace, and anything below that is "better than random". For some difficult applications, a log-loss as high as 0.4 is good. But this depends a lot on the domain!
# 
# ## Implementation
# 
# Here's an implementation on a prediction of NYC building sales: how well can be predict whether or not a unit will sell for 1+ million dollars, based on the number of units alone?

# In[ ]:


from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[ ]:


import pandas as pd
import numpy as np
sales = pd.read_csv("../input/nyc-rolling-sales.csv", index_col=0)
sales.head(3)


# In[ ]:


df = sales[['SALE PRICE', 'TOTAL UNITS']].dropna()
df['SALE PRICE'] = df['SALE PRICE'].str.strip().replace("-", np.nan)
df = df.dropna()

X = df.loc[:, 'TOTAL UNITS'].values[:, np.newaxis].astype(float)
y = df.loc[:, 'SALE PRICE'].astype(int) > 1000000

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X[:1000], y[:1000])
y_hat = clf.predict(X)


# In[ ]:


from sklearn.metrics import log_loss
log_loss(y, y_hat)


# That's some really bad performance. :D
