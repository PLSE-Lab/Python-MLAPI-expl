#!/usr/bin/env python
# coding: utf-8

# **Objective**
# 
# Based on the data from shared by the John Hopkins University, we have built a 2019-nCov Confirmed Cases Trending Dashboard. The purpose of this dashboard is to understand the outbreak situation and whether the situation is undercontrolled in specific Province.
# 
# **Usage**
# 
# The top part of the dashboard is to show gloabl infection numbers. 
# The bottom part of the dashboard focuses on the growth trends of different Province.
# 
# **Calculations**
# 
# The "Daily Confirmed Cases 1-Day Growth Rate (%) in Last 5 days" chart shows how the growth rate of top Province varies in the last 5 days which will give a sense of whether the situation is improving or worsening.
# 
# The "Confirmed Cases 2-Days Growth Rate (%) in Last 5 days" chart shows similar idea but we used "2-Days Growth Rate" to smoothen the changes. This give us a more accurate view on the situation. 
# 
# **Observation**
# 
# Most of the Province have taken measure in controlling the situation a couple days a ago (or since outbreak in the Province). For most of the Province with top growth rate, the growth rate actually drops at least half which demostrated the efforts by the Chinese Government.
# 
# However, we also see some alarming observations. From the 2-Days Growth Rate charts, we found that there are 4 Provinces showing a growing growth rate as of Feb 2, including Guangdong, Jiangsu, Anhui and Hunan. This could mean that the situation is actually worsening.
# 
# Lastly, although we see that Hubei growth rate is highest among all the Provinces (the province where Wuhan located in), the numbers shown a picture of improving situation. The 2-Days growth rate of Hubei is slowing (or constant). We hope that the situation will keep going this way.
# 
# 
# Stand-alone dashboard url: https://datastudio.google.com/s/qkKskYtQ23I

# In[ ]:


from IPython.display import IFrame
IFrame('https://datastudio.google.com/embed/reporting/c84945c0-21a5-42ee-bd79-e6e02dfa43c6/page/eHUDB', width='100%', height=1024)


# In[ ]:




