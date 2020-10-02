#!/usr/bin/env python
# coding: utf-8

# **Normal distribution math problem 2**

# In[ ]:


import scipy.stats as st
mean_normal = 1000
sd_normal = 100
probability_lt = st.norm.cdf(790,mean_normal,sd_normal)
print(probability_lt)


# **Normal distribution math problem 3**

# In[ ]:


import scipy.stats as st
mean_normal = 1000
sd_normal = 100
probability_lt_790 = st.norm.cdf(790,mean_normal,sd_normal)
probability_lt_1000 = st.norm.cdf(1000,mean_normal,sd_normal)
probability_in_between = probability_lt_1000 -probability_lt_790
print(probability_in_between)

