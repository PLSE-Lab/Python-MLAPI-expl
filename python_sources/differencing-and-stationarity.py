#!/usr/bin/env python
# coding: utf-8

# Correlations only make sense when both time series are stationary. This analysis shows two things:
# 
# 1. features are non stationary and require differencing.
# 2. correlations are different for different ids.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import kagglegym
env = kagglegym.make()
observation = env.reset()
train = observation.train


# In[ ]:


train.fillna(0, inplace=True)


# In[ ]:


gf = train.copy(True)
gf = gf.set_index('timestamp', 'id')


# Let us take technical 20 as an example and run a simple correlation.

# In[ ]:


print (np.corrcoef(train['technical_20'].values, train.y.values)[0, 1])


# Now lets plot technical 20 for a single asset. say id = 0.

# In[ ]:


import matplotlib.pyplot as plt
X = gf.loc[gf.id == train.id[0]]['technical_20'].values
Y = gf.loc[gf.id == train.id[0]]['y'].values
plt.plot(X, color='r')
plt.show()


# That does not look stationary. Now lets plot after taking the first differential of technical 20 by asset Id 0

# In[ ]:


X = np.diff(X)
plt.plot(X, color='r')


# Still does not look stationary. Lets difference again.

# In[ ]:


X = np.diff(X)
plt.plot(X)
plt.show()


# Now lets compute the correlation between the two times differenced X and Y values.

# In[ ]:


print (np.corrcoef(X, Y[2:])[0, 1])


# **That is a whopping -18%**

# But what about other assets? Let us try a random 47th asset.

# In[ ]:


X = gf.loc[gf.id == train.id[47]]['technical_20'].values
Y = gf.loc[gf.id == train.id[47]]['y'].values
X = np.diff(X)
X = np.diff(X)
print (np.corrcoef(X, Y[2:])[0, 1])


# This is better than running correlations on non-stationary data but still is only positive 4%.

# **CONCLUSION:**
# 
# Correlations on un-differenced data are spurious and make no sense. You have to difference to find stationary series before looking for correlations. Fortunately train.y (appears to be asset returns) are already stationary.
# 
# Secondly correlations are not same across asset ids. Each asset has a different correlation to features.
# 
# Hope this helps. Now lets get cracking this challenge.

# *If you like this analysis please upvote. Else let me know if I have misunderstood something.*
