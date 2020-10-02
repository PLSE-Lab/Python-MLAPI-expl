#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import datetime
import matplotlib.pyplot as plt
import fix_yahoo_finance as yf 
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (14, 8) # Change the size of plots


# # Bond Market
# As rates go higher, it becomes more difficult for businesses to find a loan from the bank. If company XYZ wants to get a 100,000 loan with a 0.01% interest rate, they must have an expected return at least equal to 101,000. Therefore, higher interest rates are bad for business confidence.
# 
# Recently in economic-related news, you may have heard about how President Trump, banks, hedge fund managers, etc. are complaining about the decisions of the Federal Reserve. Interest rates have been at historic lows since the global financial crisis in 2007-2008. There has been a process of the Fed going through quantiative easing to try and 'fix' the aftermath of the global economy turning downwards. Now that the economy has been fixing itself, the Fed has been looking to raise rates again so that the economy doesn't spiral out of control in the other direction. The Federal Reserve, although having members elected, traditionally acts independently of politics so that it is free to make rational decisions.
# 
# There seems to be an illusion that the Fed itself, who decides interest rates, is also the one somehow acting as the mastermind capable of controlling where the global markets are headed. With these large political and financial figures saying that the Fed 'should' or 'shouldn't' do something, it is beneficial to use some plotting to explain exactly what the interest rates have been doing for the past few decades.
# 
# It becomes quite clear, that the interest rates follow the 90 day treasury bill market. If you don't know how the treasury bill or bonds work, it is where the government sells certificates at a certain rate, and promises to pay others back a percentage of the loan over time with interest in the end.

# In[ ]:


start = '1990-01-01' # Beginning of FOMC target rate data
end = '2018-11-14'


# In[ ]:


fomc = pd.read_csv('../input/fomc1.csv') # Load/format FOMC target data
fomc.set_index(pd.DatetimeIndex(fomc.Date), inplace=True)
fomc = fomc.resample('D', fill_method='pad').drop(['Date'], axis=1)


# In[ ]:


tbill13 = yf.download('^IRX', start, end) # Load tbill, bond data
ty5 = yf.download('^FVX', start, end)
bond10yr = yf.download('^TNX', start, end)
ty30 = yf.download('^TYX', start, end)


# In[ ]:


bonds = pd.DataFrame({'5 year': ty5['Adj Close'], # plot
                      '90 day': tbill13['Adj Close'],
                      '10 year': bond10yr['Adj Close'],
                      'FOMC rates': fomc['Level'],
                      '30 year': ty30['Adj Close']})
bonds.plot(grid = True)
plt.show()


# It may become apparent that the Fed acts in a lagging manner, following whatever trend the 90 day treasury bill is headed. This is quite a contradicting narrative to what is heard on the economic-related news. Inspiration for this data come from: https://www.socionomics.net/2016/11/article-central-banks-do-not-control-interest-rates-in-europe-either/ . It is not my original idea to plot the correlation between the two instruments.
