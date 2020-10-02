#!/usr/bin/env python
# coding: utf-8

# My hypothesis is that more popular kernels are associated with lower private leaderboard scores relative to public scores.  The idea, loosely speaking, is that popular kernels are selected based in part on public leaderbaord scores and therefore more likely to be overfit.  In this very basic analysis, support for the hypothesis is quite limited, and so far not statistically significant (in part because the data set is too small and doesn't provide enough power).

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm


# In[ ]:


df = pd.read_csv("../input/porto-seguro-public-kernel-results/portoKernels.csv",index_col=0)
df.head()


# In[ ]:


df.set_index('votes').drop(['reported public score','adjusted'],axis=1).sort_index(ascending=False)


# A few quick OLS runs

# In[ ]:


X = df.rename(columns={'public score':'pubscore','private score':'privscore'})
sm.ols(formula="difference ~ pubscore+votes", data=X).fit().summary()


# In[ ]:


sm.ols(formula="difference ~ votes", data=X).fit().summary()


# In[ ]:


sm.ols(formula="privscore ~ pubscore+votes", data=X).fit().summary()


# Not statistically significant, but the estimates suggest that each vote reduces relative private leaderboard performance by about one point in the fifth decimal place.

# Let's look at the relationship graphically.

# In[ ]:


plt.scatter(X.votes,X.difference)
plt.axis([0, 250, -.003, .012])
plt.show()


# Certainly doesn't look like a strong relationship, but *maybe* a weak nonlinear relationship, where kernels with more than, say, 25 votes tend to do a little worse in the shake-up.

# I'm also curious about the extent to which higher-scoring public kernels tend to lose in the shake-up.  To the extent that this effect exists, it could reflect overfitting, but it could also just be regression to the mean, given that there is some randomness in both public and private scores.

# In[ ]:


plt.scatter(X.pubscore,X.difference)
plt.axis([0, .3, -.003, .012])
plt.show()


# In[ ]:


# More detail on the right side
plt.scatter(X.pubscore,X.difference)
plt.axis([.2, .3, -.003, .012])
plt.show()


# Again, not a strong relationship.

# Let's look directly at the relationship between public and private scores.

# In[ ]:


sm.ols(formula="privscore ~ pubscore", data=X).fit().summary()


# In[ ]:


plt.scatter(X.pubscore,X.privscore)
plt.axis([0, .3, 0, .3])
plt.show()


# In[ ]:


# More detail on the right side
plt.scatter(X.pubscore,X.privscore)
plt.axis([.2, .3, .2, .3])
plt.show()


# In[ ]:


# Even more detail on the far right side
plt.scatter(X.pubscore,X.privscore)
plt.axis([.270, .292, .270, .292])
plt.show()


# In[ ]:


# Yet even more detail on the very far right side
plt.scatter(X.pubscore,X.privscore)
plt.axis([.278, .288, .278, .291])
plt.show()


# In[ ]:


# Yet still even more detail on the extreme far right side
plt.scatter(X.pubscore,X.privscore)
plt.axis([.282, .288, .282, .291])
plt.show()


# I'm really pretty impressed:  the public test data were quite good at sorting true performance, even when there was "crowdsourced overfitting" (Bojan's term IIRC) involved.  I'm going to say that there is really no evidence here that overfitting by public kernels is a major problem, at least in this particular competition.  I'd still want to downweight public kernels in my ensembles, in part because the data are still reaonably consistent with my prior belief that there is some overfitting, and in part just becuase of regression to the mean (since the crowd will choose kernels with good results, which are more likely to regress downward).

# In[ ]:




