#!/usr/bin/env python
# coding: utf-8

# # Open Boundary Condition
# ![The edge of a flat world](http://static.tvtropes.org/pmwiki/pub/images/rsz_world-flat_5884.jpg)
# 
# The data are generated from a model with open boundary condition.
# Let see the histogram of checkins along axe x and close to the border x=0. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16,5)


# In[ ]:


train = pd.read_csv('../input/train.csv', index_col='row_id')


# In[ ]:


train[train.x<0.1].x.hist(bins=100)

plt.xlabel('x')
plt.ylabel('Number of checkins');


# The pictures for all 4 borders is similar ([A 3D plot by Namra](https://www.kaggle.com/namra42/facebook-v-predicting-check-ins/gridwise-open-boundary)). I think the reason is: 
# 
# The reported position of a device (x,y), 
# might be different from its true position (x_t, y_t) because of the errors. 
# Now assume that the probability of having a value lesser or greater that its true value 
# is equal, P(x > x_t | x_t) = P(x < x_t | x_t). 
# For any point that is far from border, we should get an almost flat density P(x) = const. 
# But on the border the story is different. 
# 
# If the true position is on the border, 
# and the measured position is inside the border, 
# then we see that device, but if the observed is outside of the border, 
# then we have a loss, its coordinates are not reported. 
# Beyond the borders there is only void so there is nothing to compensate the loss. 
# We have lower density of checkins at the borders. 
# The density increases as we go farther. 
