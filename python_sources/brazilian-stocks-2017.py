#!/usr/bin/env python
# coding: utf-8

# # Stock Analysis for 2017
# 
# I have a group of companies which I'm looking closer in the last few years.
# They are solid companies or emerging startups with high potential.
# This report is the first look at this portfolio.
# 
# 
# ## Setup
# 
# The first thing to do is to import some required libraries:

# In[ ]:


import pandas_datareader as api
import matplotlib.pyplot as plt
import numpy as np
import datetime
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Interval
# 
# Let's select an interval from 1st jan/2017 to 1st jan/2018:

# In[ ]:


start = datetime.datetime(2017,1,1)
end = datetime.datetime(2018,1,1)


# ### List of Companies:
# 
# BBAS3, BBDC4, BRML3, BTOW3, CRFB3, GOLL4, ITUB4, MOVI3, NFLX34, ORCL34, PETR4, SANB11, SEER3, VALE3

# In[ ]:


stocks_prices = {}
my_stocks = ["BBAS3", "BBDC4", "BRML3", "BTOW3", "CRFB3", "GOLL4", 
             "ITUB4", "MOVI3", "NFLX34", "ORCL34", "PETR4", "SANB11", 
             "SEER3", "VALE3"]


# ### API Query
# 
# I am using the Google Stock API and saving the results to a dictionary.

# In[ ]:


# The regular API query via Pandas DataReader is not working on Kaggle
# due to Internect restrictions.
# I've solved the problem by creating an offline numpy dump of the dataset.

# Uncomment these lines to run on your computer:
# for stock in my_stocks:
#    stocks_prices[stock] = api.DataReader(stock, "google", start, end)

# Comment these lines to run on your computer:
stocks_prices = np.load('../input/offline_2017_stocks.np.npy').item()


# It is important to count the entries and check if some problem is ocurring during the API request.

# In[ ]:


for key in stocks_prices:
    print("Shape of ", key, ": ", len(stocks_prices[key]))


# ## Graphical Analysis
# 
# The following is a plot of all stocks listed above.

# In[ ]:


for stock in stocks_prices:
    plt.figure(figsize=[15,5])
    plt.title('Price for %s in 2017' % stock)
    plt.plot(stocks_prices[stock]['Close'])
    plt.grid()
    plt.xlabel("2017")
    plt.ylabel("R$")
    plt.show()


# ## Facts
# 
# Looking to most graphics above it is possible to find a strange behavior in May/2017.
# 
# What happened?
# 
# I will choose some stocks of big banks in Brazil and see if it is happening.

# In[ ]:


plt.figure(figsize=[18,6])
plt.plot(stocks_prices['BBAS3']['Close'],'y-', label="Brasil")
plt.plot(stocks_prices['BBDC4']['Close'],'r-', label="Bradesco")
plt.plot(stocks_prices['ITUB4']['Close'],'b-', label="Itau")
plt.plot(stocks_prices['SANB11']['Close'],'g-', label="Santander")
plt.grid()
plt.legend()
plt.xlabel("2017")
plt.ylabel("R$")
plt.show()


# IT IS HAPPENING! What is causing this price fall?
# 
# A good idea is to look at the daily report for one affected stock and find the exact moment it is happenning.

# In[ ]:


bbas = stocks_prices['BBAS3']
bbas_may = bbas[bbas.axes[0].month.isin(['05'])]


# In[ ]:


plt.figure(figsize=[18,6])
plt.plot(bbas_may['Volume'],'g-')
plt.grid()
plt.show()
plt.figure(figsize=[18,6])
plt.plot(bbas_may['Close'],'r-')
plt.grid()
plt.show()


# In[ ]:


bbas_may


# The daily reports and both (Volume and Close) graphics are showing a big event impacting the negotiations on May 18th. Let's search the internet what caused such problem.
# 
# A quick Google Search for "News May 18th 2017 Brazil Bovespa" reveal the political scandal:
# 
#  - https://www.cnbc.com/2017/05/18/this-brazil-stocks-etf-is-crashing-more-than-13-percent-on-an-emerging-political-scandal.html
#  - http://money.cnn.com/2017/05/18/investing/brazil-stock-market-temer/index.html
#  - https://g1.globo.com/economia/mercados/noticia/bovespa-fecha-em-forte-queda-de-olho-em-denuncias-sobre-temer.ghtml
#  - https://g1.globo.com/economia/mercados/noticia/bovespa-180517.ghtml
#  
# It looks like the political scandals affected many segments: banks, industry, retail and transportation.
# 

# ##### Wesley Rodrigues da Silva - 2018
