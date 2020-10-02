#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file
import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting
from mpl_toolkits.mplot3d import Axes3D

# Input data files are available in the "../input/" directory.

import os


# In[ ]:


efw = pd.read_csv("../input/efw_cc.csv")


# The index measures the degree of economic freedom present in five major areas: #1. Size of Government, #2. Legal System and Property Rights, #3. Sound Money #4. Freedom to Trade Internationally, and #5. Regulation of credit, labor, and business. Each component (and sub-component) is placed on a scale from 0 to 10, where 10 represents more freedom. 
# 
# When comparing the levels of economic freedom for years 1980 and 2016, we see that the least free country in 1980 had a score of 2.66, whereas the least free country in 2016 scored 2.88. The max scores for 1980 and 2016 were 8.73 and 8.97, respectively. The mean also increased between 1980 and 2016 from 5.61 to 6.80. Note that in 1980 only 113 countries were analyzed. In 2016, 162 countries composed the index.

# In[ ]:


#Slice data frame for the years we want to study. We will use these later.
efw_2016 = efw.loc[efw['year'] == 2016]
efw_1980 = efw.loc[efw['year'] == 1980]

#Select main categories from sliced data frames for summary statistics
efw2016_mc = efw_2016[["ECONOMIC FREEDOM", "rank", "1_size_government", "2_property_rights", "3_sound_money", "4_trade", "5_regulation"]]
efw1980_mc = efw_1980[["ECONOMIC FREEDOM", "rank", "1_size_government", "2_property_rights", "3_sound_money", "4_trade", "5_regulation"]]


# In[ ]:


print("Summary statistics for 2016")
display(efw2016_mc.describe())
print("Summary statistics for 1980")
display(efw1980_mc.describe())


# A similar conclusion can be drawn from the kernel density plot below. In 1980, roughly 50% of the countries' scores ranged between 4 and 6. In 2016, more than half of the countries scored between 6 and 8 in economic freedom. 

# In[ ]:


sns.kdeplot(efw_2016["ECONOMIC FREEDOM"], label="2016", shade=True)
sns.kdeplot(efw_1980["ECONOMIC FREEDOM"], label="1980", shade=True)
plt.legend()
plt.title("Economic Freedom, 1980 and 2016")
_ = plt.xlabel("Economic Freedom score")


# Here's the evolution of economic freedom and its components over time.

# In[ ]:


efw_gb = efw.groupby("year").mean()
_ = efw_gb.plot(y=["ECONOMIC FREEDOM", "1_size_government", "2_property_rights", "3_sound_money", "4_trade", "5_regulation"], figsize = (10,10), subplots=True)
_ = plt.xticks(rotation=360)


# How do the economic freedom areas correlate to each other?

# In[ ]:


efw2016_corr = efw_2016[["1_size_government", "2_property_rights", "3_sound_money", "4_trade", "5_regulation"]]
sns.heatmap(efw2016_corr.corr(), square=True, cmap='RdYlGn')
plt.show()


# How has economic freedom and its components changed over time? Economic freedom increased by 0.83 between 1970 and 2016. Trade and Regulation experienced the biggest increases, whereas Property Rights was the only area that saw a decline between that same period of time.

# In[ ]:


efw_mc = efw[["year","ECONOMIC FREEDOM", "1_size_government", "2_property_rights", "3_sound_money", "4_trade", "5_regulation"]]
efw_gb = efw_mc.groupby("year").mean()

efw_gb.loc[2016] - efw_gb.loc[1970]


# In[ ]:




