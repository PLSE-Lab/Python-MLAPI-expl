#!/usr/bin/env python
# coding: utf-8

# # Who might be sharing or selling my personal data?
# 
# Enterprises has to public disclosure what they do with personal data through their privacy policy, as way to getting compliance by GDPR, CCPA, LGPD and other laws. Privacy policy is a big and boring document to read and most users don't read it at all. After a while, people start to complain when receives spam and calls offering products from unknowns. People are becoming more aware about privacy. This project shows: 
# 1. Discover which personal information companies are collecting about you, by selecting services that you commonly use; 
# 2. Discover who might be collecting personal data from you, by selecting which type of personal information you care about.
# 
# **Analysis:** Scrapping top ranking website URL, then find privacy policy from each site, read it and then performs NLP to extract every type of personal information collected, transform into a database and create beautiful and useful visualisations that anyone can easily understand.
# 
# **Data:** private policies from top websites. The list of most accessed website is from alexa ranking. This first version is limited by few companies and only english language policies. I aim to increase this list to more websites during the bootcamp, including other countries and languages.
# 
# **Source:**
# * https://www.alexa.com/topsites
# * https://usableprivacy.org/data/

# In[ ]:


import pandas as pd
urls = pd.read_csv('../input/april_2018_policies.csv')
urls.head()


# In[ ]:


urls['Policy Sources'][1]


# In[ ]:




