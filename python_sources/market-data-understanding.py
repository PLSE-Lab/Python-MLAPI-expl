#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from kaggle.competitions import twosigmanews
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') # nice theme for plotting

env = twosigmanews.make_env() # load env
market_data = env.get_training_data()[0] # load only market data
# save the dates for later purpose
dates = pd.to_datetime(market_data['time'].unique())


# ## Universe
# 
# #### How many available assets do we have at any point in time?

# In[ ]:


import matplotlib.pyplot as plt
market_data.groupby('time').count()['universe'].plot(figsize=(12, 5) ,linewidth=2)
print("There are {} unique investable assets in the whole history.".format(market_data['assetName'].unique().shape[0]))


# According to the description, we are whithin an US stock only-listed universe. However, the number of available companies at any point in time seems to be market dependant. Even if the number of total assets is 3511, there must be an algorithm that pre-selects the universe, converting it into the investment universe. I can only think in 3 reasons why this may be happening:
# * [Survivorship bias](https://en.wikipedia.org/wiki/Survivorship_bias): Stocks usually trade on a market but are part of an index/market reference/benchmark. There are ins and outs of an index due to market capitalization requirements and M&A events. Therefore, the number of available stocks may change over time. 
# * Data filtering: Some filtering algorithm may be run before creating the investment universe. Think of a stituation in which we had no price for a specific security in the past week. What quantitate algorithm is going accept NaN values? The price time series may have been involved in a, let's say, data quality test.
# * Market movements: Looking to the previous picture, there's a [big drop round the second part of 2015, specifically around August](https://en.wikipedia.org/wiki/2015%E2%80%9316_stock_market_selloff). The thing is, another filtering algorithm regarding the methodology of the quantitative team may have been applied in the whole dataset to avoid exposure to downside asset movements. 

# ## Prices & Returns
# 
# Prices can be raw or dividend-adjusted. When a company is delivering dividends, the price of the serie falls in the same amoun of the declared dividend since the price series has that information in the price value. Moreover, when companies want to increase their capital, they do splits. This is, they increment/decrease the number of shares without altering their market capitalization, so their price decrease/increase proportionally to mantain that market cap value. How do we now this? Again, look at Apple's close price:

# In[ ]:


close_price_df = market_data.pivot_table(index='time', values='close', columns='assetCode')
close_price_df['AAPL.O'].plot(linewidth=2, figsize=(12, 5))


# As you see, that big jump comes from the fact thay they made a 7:1 split. Click [here](https://www.stocksplithistory.com/apple/) for more info. 
# 
# Luckily, the returns are adjusted and take into account all this effects:

# In[ ]:


ticker = 'AAPL.O'
close_1d_returns_raw = market_data.pivot_table(index='time', values='returnsClosePrevRaw1', columns='assetCode')
close_1d_returns_adj = market_data.pivot_table(index='time', values='returnsClosePrevMktres1', columns='assetCode')
tmp_r = pd.concat([close_1d_returns_raw[ticker],close_1d_returns_adj[ticker]], 1)
tmp_r.columns = ['1 day close-to-close raw returns'.format(ticker), '1 day close-to-close market residualised {} returns'.format(ticker)]
tmp_r.plot(linewidth=1, alpha=0.7, figsize=(12, 5))


# If the split hadn't got taken into account, there would be a -700% return for that day! 
# 
# #### Returns
# One can calculate returns in this way:
# \begin{equation*}
# R_t = \frac {P_t - P_{t-1}} {P_{t-1}}
# \end{equation*}
# Or this way:
# \begin{equation*}
# R_t = ln \frac {P_t} {P_{t-1}}
# \end{equation*}
# The consequences are notorious. Generally, log computed returns tend to be more homocedastic. [Check this paper for further info](https://poseidon01.ssrn.com/delivery.php?ID=680074082121113031003082022124093113005035046078090017099117006064087094108113031077114007007115059023062065027092118068017093121012037073045075108071065116119067073007064097020094010111073091092119025001072016090097030112118023106121091127009088083&EXT=pdf). How are the returns computed then?

# In[ ]:


import numpy as np
r = tmp_r.iloc[:, 0]
cum_d = (np.cumprod(1 + r) - 1)
cum_l = r.cumsum()
cum_ret = pd.concat([cum_d, cum_l], 1)
cum_ret.columns = ['pct change {} returns'.format(ticker), 'log {} returns'.format(ticker)]
cum_ret.plot(linewidth=2, alpha=1, figsize=(12, 5))


# Well the blue series looks like more like the Apple cumulative return! So the returns are calculated with the percentual change. This will be key when modelling in the future since we all know more betther about the properties of financial returns

# # More work to come!!

# In[ ]:




