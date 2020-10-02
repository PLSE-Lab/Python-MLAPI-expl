#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pytrends')
import pytrends
#https://github.com/GeneralMills/pytrends/blob/master/examples/example.py
from pytrends.request import TrendReq
pytrend = TrendReq()


# In[ ]:


from datetime import date
today = date.today()


# In[ ]:


pytrend.trending_searches(pn='india') 


# In[ ]:


pytrend.suggestions('Machine Learning')


# In[ ]:


_=pytrend.get_historical_interest(['Machine learning'], year_start=2018, month_start=1, day_start=1, hour_start=0, year_end=2018, month_end=2, day_end=1, hour_end=0, cat=0, geo='', gprop='', sleep=0)['Machine learning'].plot()


# In[ ]:


pytrend.categories()

