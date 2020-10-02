#!/usr/bin/env python
# coding: utf-8

# # Overview

# ### What is the goal of Project [COVIEWED](https://www.coviewed.org/)?

# Project [COVIEWED](https://www.coviewed.org/)'s aim is to fight against misinformation on the web regarding the recent coronavirus pandemic / covid-19 outbreak. 
# 
# To achieve this, we collect different types of claims from web sources with supporting or attacking evidence. 
# 
# This information is used to train a machine learning classifier. 
# 
# Once trained, we plan to release a browser extension that highlights potential true/false claims on a web page to assist users in their information gathering process.

# ### Organization

# We have a public [Trello Board](https://trello.com/invite/b/jk00CW3u/9985740815b585156aaa22978a3067df/project-coviewed) and our project is completely open source and available on [Github](https://github.com/COVIEWED).

# # Set-Up

# In[ ]:


get_ipython().system('pwd')


# In[ ]:


get_ipython().system('ls -l')


# In[ ]:


get_ipython().system('git clone https://github.com/COVIEWED/coviewed_web_scraping')


# In[ ]:


get_ipython().system('pip install -r coviewed_web_scraping/requirements.txt')


# In[ ]:


get_ipython().system('git clone https://github.com/COVIEWED/coviewed_data_collection')


# In[ ]:


get_ipython().system('pip install -r coviewed_data_collection/requirements.txt')


#     The earliest entry is from 2020-01-20

# In[ ]:


get_ipython().system('cd coviewed_data_collection/ && python3 src/get_reddit_submission_links.py -a=2020-03-28T23:59:00 -b=2020-03-29T00:00:00 -s=Coronavirus --verbose')


# In[ ]:


get_ipython().system('ls coviewed_data_collection/data/*.tsv')


# In[ ]:


get_ipython().system('cd coviewed_data_collection/ && python3 src/get_reddit_submission_texts.py -s=Coronavirus --verbose')


# In[ ]:


get_ipython().system('ls coviewed_data_collection/data/*.txt')


# In[ ]:


get_ipython().system('cp coviewed_data_collection/data/news_urls.txt coviewed_web_scraping/data/')


# In[ ]:


get_ipython().system('ls coviewed_web_scraping/data/*.txt')


# In[ ]:


get_ipython().system('wc -l coviewed_web_scraping/data/news_urls.txt')


# In[ ]:


get_ipython().system('cd coviewed_web_scraping/ && sh run.sh')


# In[ ]:


get_ipython().system('ls coviewed_web_scraping/data | grep .txt | wc -l')


# In[ ]:




