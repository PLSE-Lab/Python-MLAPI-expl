#!/usr/bin/env python
# coding: utf-8

# This short kaggle report presents a data visualization of frequent topics highlighted in an online feedback conducted by New Naratif

# In[ ]:


import nltk
from nltk.corpus import stopwords
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
RESPONSES = '../input/sg-public-feedback/CitizensAgenda_201909_English.csv'


# Top Singapore issues, online survey feedback from New Naratif : Citizens Agenda.
# Main article published 04 Sep 2019 : https://newnaratif.com/research/the-28-most-important-issues-facing-singapore/
# 
# Shortlist of issues (28) hand selected by the article authors
# 
# Cost of Living & Poverty
# 
# CPF, Insurance, Welfare
# 
# Demography
# 
# Destructive Politics
# 
# Discrimination
# 
# Drugs/Addictive Substances
# 
# Economy and Jobs
# 
# Education
# 
# Electoral and Parliamentary Reform
# 
# Environment/Climate Change
# 
# Foreign Policy
# 
# Healthcare
# 
# Housing/HDB
# 
# Human Rights and Civil Liberties
# 
# Immigration and Refugees
# 
# Inequality
# 
# Infrastructure and Transport
# 
# Language teaching
# 
# Leadership and Politics
# 
# National Service
# 
# Religion
# 
# Security
# 
# Society
# 
# Tax
# 
# Technology, AI, and the Fourth Industrial Revolution
# 
# The Future of Singapore and Singaporean Identity
# 
# Transparency and Accountability

# In[ ]:


#read in the raw data
df = pd.read_csv(RESPONSES,encoding="ISO-8859-1")


# Here a subset of the topics is hand-defined based on keyword matches.  This is approximate and can be refined manually and re-run to give different answers.

# In[ ]:


topics = {'climate change':['climate change','climate crisis','environment'],
         '377a':['377a','lgbt','gay','queer'],
         'pofma':['pofma','fake news','freedom of speech'],
         'cpf':['cpf','retirement','gic','temasek'],
         'education':['education'],
         'healthcare':['healthcare','medishield'],
         'democracy':['democracy','transparency','nepotism'],
         'economy':['economy','jobs','income','inflation','wage'],
         'cost of living':['cost of living'],
          'inequality':['inequality'],
          'race':['racial','racism','races','ethnicity','religion','religious'],
         }


# In[ ]:


topic_count = {}
for row in df.iterrows():
    for t in topics:
        tk_matches = [1 for tk in topics[t] if tk in row[1].response.lower()]
        if len(tk_matches)>0:
            if t in topic_count: 
                topic_count[t] +=1
            else:
                topic_count[t] = 1


# In[ ]:


tp = pd.DataFrame({'count':pd.Series(topic_count)}).sort_values('count')
tp['count'] = tp['count']/sum(tp['count'])


# In[ ]:


import matplotlib.pyplot as plt
tp.plot.barh()
plt.title('top Singapore issues')


# Results show top three issues are economy, CPF and transparency of government, followed by income inequality, climate change and repeal of proposition 377A 
