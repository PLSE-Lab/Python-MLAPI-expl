#!/usr/bin/env python
# coding: utf-8

# # ONE LINE PYTHON SERIES
# 
# One Line Python is beginners guide for shortest way to learn python tips and tricks.
# 
# This Series show you how to analyze data with minimum effort. You don't need to know high level python programming for good looking notebooks. Just follow series and use what you have learned in your notebooks.
# 
# **I hope you find this notebook helpful and some <font color="red"><b>UPVOTES</b></font> would be very much appreciated.**
# 
# ### This Notebook includes <font color="green"><b>Web Scraping for Beginners</b></font>
# 
# Actually web scrape is hard topic and need andvanced skills. (I'll be add on another noteboook Advanced Techniques for web scraping)
# 
# But in some cases you can do it with an easy way
# 
# There are two rules to use that code below.
# 
# * First rule ise the web page that you copied link, has to allow scraping. (some sites using ekstra security for scraping)
# * Second rule is the web page that you copied link must have <tr><td> flags. (<tr><td> flags using for creating table in webpage)
# 
# 
# 
# Follow the below code
# 
# 
# 
# 
# See you on other one line Python series
# 
# ## [One Line Python - Part 1 - Pandas Profiling](https://www.kaggle.com/medyasun/one-line-python-part-1-pandas-profiling)
# ## [One Line Python - Part 2 - Image Link](https://www.kaggle.com/medyasun/one-line-python-part-2-image-link)

# In[ ]:


import pandas as pd
read_html=pd.read_html("https://www.imdb.com/chart/top/")   # pandas powerful code line for reading tables in html pages
data=read_html[0]                                           # read_html returns list item. First item of that list returns pandas dataframe
data

