#!/usr/bin/env python
# coding: utf-8

# # Wald Confidence Intervals with Iowa Liquor Sales
# 
# In [a previous notebook](https://www.kaggle.com/residentmario/bootstrapping-and-cis-with-veteran-suicides) I described how to obtain a confidence interval for an estimate of an interval variable.
# 
# In this notebook, I'll examine the Wald Confidence Interval, the simplest method for generating a confidence interval for binomial trails.
# 
# A Bernoulli trial occurs when you have a sequence of data points which can only take on two values: say, 0 and 1. Examples of Bernoulli trials include which of two different players will be the one to win a tennis match, or predicting whether or not a train is going to arrive in the next five minutes of waiting. Either something happens, or it doesn't happen. Usually 1 is denotated "success" and 0 "failure".
# 
# ## Derivation
# 
# Let $p$ be the probability of success. Conversely, $q = 1 - p$ is the probability of failure. Let $X$ be the number of successes in $n$ trials.
# 
# Our goal is to understand $p$. We will do this by constructing an estimator on $p$, $\hat{p}$, using the observed properties of our model.
# 
# $E[X]$ is the expectation of $X$: the expected number of succeses in $n$ trials. From there we have that:
# 
# $$E[\frac{X}{n}] = \frac{1}{n}(np) = p$$
# 
# In other words, the expectation of the number of succeses over the number of trials is $p$, the underlying probability of success.
# 
# Similarly, $Var(\frac{X}{n})$, the variance of the estimate, is given by:
# 
# $$Var[\frac{X}{n}] = \left(\frac{1}{n}\right)^2 V[X] = \frac{npq}{n^2} = \frac{pq}{n}$$
# 
# The central limit theorem states that since $E[\hat{p} = \frac{X}{n}] = p$, with a large enough number of samples (~$n \geq 30$), the error committed by $\hat{p}$ will be normally distributed.
# 
# (to see why this happens in more detail, refer to [the previous notebook](https://www.kaggle.com/residentmario/bootstrapping-and-cis-with-veteran-suicides))
# 
# We can normalize $\hat{p}$ to arrive at a confidence interval.
# 
# $$P\left( -z_{\alpha/2} < \frac{\hat{p} - p}{\sqrt{pq/n}} < z_{\alpha / 2} \right) = 1 - \alpha$$
# 
# Here $z$ is the standardized z-score for how confident we want to be; since this is a two-sided interval for given confidence $\alpha$ (e.g. $\alpha$=0.95) we need to half the interval.
# 
# The boundaries for the confidence interval will be the endpoints, which will be:
# 
# $$\frac{\hat{p} - p}{\sqrt{pq/n}} = \pm z_{\alpha / 2}$$
# 
# Hence:
# 
# $$p = \hat{p} \pm z_{\alpha/2}\sqrt{p(1-p)/n}$$
# 
# The "Wald approximation" is the introduction of additional error into this result by swapping out $p$ (an underlying statistic) for $\hat{p}$, an estimate of $\hat{p}$:
# 
# $$p = \hat{p} \pm z_{\alpha / 2}\sqrt{\hat{p}(1 - \hat{p})/n}$$

# ## Implementation

# In[ ]:


import scipy.stats as st
import numpy as np

def wald_confidence_interval(X, c):
    n = X.shape[0]
    
    p_hat = X.astype(int).sum() / n
    z_score = st.norm.ppf(1 - ((1 - c) / 2))

    additive_part = z_score * np.sqrt(p_hat * (1 - p_hat) / n)
    
    return (p_hat - additive_part, p_hat + additive_part)


# ## Application
# 
# Let's apply the Wald CI to a somewhat random but cute problem: given one million liquor sales in the state of Iowa (one mil. to keep the computation time decent), how well can we estimate what the probability of someone making an alcohol purchase on Christmas Day?

# In[ ]:


import pandas as pd
sales = pd.read_csv("../input/Iowa_Liquor_Sales.csv")


# In[ ]:


_sales = (sales
     .head(1000000)
     .assign(n=0)
     .groupby('Date')
     .count()
     .n
     .to_frame()
     .reset_index()
     .pipe(lambda df: df.assign(Date=pd.to_datetime(df.Date)))
     .pipe(lambda df: df.assign(Day=df.Date.dt.dayofyear))
     .groupby('Day')
     .mean()
)

christmas_day_sales = _sales.loc[359, 'n']
all_sales = _sales.n.round().sum()


# In[ ]:


is_christmas_sale = np.array([True]*int(christmas_day_sales) + [False]*(1000000 - int(christmas_day_sales)))


# In[ ]:


wald_confidence_interval(is_christmas_sale, 0.95)


# We are 95% confident that the true proportion of alcohol sales that occur on Christmas Day is between ~0.00168 and ~0.00185.
# 
# Note that this means that, as expected, the amount of alcohol that gets purchased on Christmas is *significantly less* than the amount that gets purchased on an average day of the year:

# In[ ]:


1/365

