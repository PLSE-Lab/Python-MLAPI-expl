#!/usr/bin/env python
# coding: utf-8

# #Classification of Series
# I am not a math expert. I am trying to find clever tricks to solve this problem. We need to divide the series into bins for ML to work better on them.
#  
# There are different kind of series. We will try to categorize them into groups. Lets start with simple classification: oscillating (OS) or non oscillating (NOS). Other way to classify is to check if it is increasing (I), decreasing (D), or range bound (RB).
# 
# We can further classify the a series as arthematic series (AS) or exponential series (ES). We can trian a different machine learning algorithm for each type of series and hopefully be more successful.
# 
# Let us read the data and examine it.

# In[ ]:


# Importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# read train data
data = pd.read_csv("../input/train.csv")

# Convert sequence into list of integer lists (convenient for accessing)
seqs = data['Sequence'].tolist()
seqsL = [list(map(int, x.split(","))) for x in seqs]


# Let us see how the first series looks:

# In[ ]:


series = seqsL[0]
print(series)


# The first series is clearly an exponential series (ES), non oscillating (NOS) and increasing (I). Let us divide each number by previous number and see how the series looks.

# In[ ]:


divSeries = [float(n)/m for n, m in zip(series[1:], series[:-1])]
plt.plot(divSeries)
plt.show()


# The above series is also exponential. Here we are not trying to find the next number in the sequence but to just classify the series. It is clear that if the s(i+1)/s(i) > 1 then the series is (ES) and (I). If  s(i+1)/s(i) <= -1 then it is (OS) and (ES). (AS) can also have ratio >1 but it tends to be closer to 1. When the series size is limited we can't be really sure. For (AS) the difference play much important role. 
# 
# Let us now see the second series
# 

# In[ ]:


series = seqsL[1]
print(series)
plt.plot(series)
plt.show()


# The graph above shows the series is oscillating and growing exponentially. How can we get that from the series? Let us take both differences and ratio.

# In[ ]:


diffSeries = [n - m for n, m in zip(series[1:], series[:-1])]
divSeries = [float(n)/m for n, m in zip(series[1:], series[:-1])]

