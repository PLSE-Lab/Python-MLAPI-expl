#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Run this cell to pip install statnnot
get_ipython().system('pip install statannot')


# In[ ]:





# In[ ]:


# Imports and initializations

import pandas as pd
import seaborn as sns

# for stattanot, make sure you 
from statannot import add_stat_annotation
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

exchange_df = pd.read_csv('../input/coingecko-vs-coinmarketcap-data/May_CMC_CG_Combo.csv')

matplotlib.__version__


# There are 4 features relating to trading pairs:
# 
# 1) Red Pairs (Percentage of pairs deemed high risk by CoinGecko)
# 2) Yellow Pairs (Percentage of pairs deemed Medium risk by CoinGecko)
# 3) Green Pairs (Percentage of pairs deemed Low risk by CoinGecko)
# 4) Unknown Pairs (Percentage of pairs with unknown risk)
# 
# To make things a little easier, we will bin red, yelllow, and unknown together into one
# 
# Let's take a look at how our binned "bad pairs" relate with other things

# In[ ]:


exchange_df['Bad_Pairs'] = exchange_df['Unknown_Pairs'] + exchange_df['Yellow_Pairs'] + exchange_df['Red_Pairs']


# Heatmaps are great because we can get a quick overlook of all the attributes and their effects on each other

# In[ ]:


# Get numeric df
numeric_df = exchange_df._get_numeric_data()
# Clean out estimated reserves and regulatory compliance
numeric_df = numeric_df.drop(columns = ['Estimated_Reserves', 'Regulatory_Compliance'])

# Get correlation matrix
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(10,10))  
sns.heatmap(corr, annot = True, linewidths=.5, ax = ax)


# Density plot of numeric values? We'll do this to get an idea of the distribution of different values.

# In[ ]:


num_cols = numeric_df.columns
fig, ax = plt.subplots(3,4,figsize=(22,10))
for i, col in enumerate(num_cols):
    plt.subplot(3,5,i+1)
    plt.xlabel(col, fontsize=9)
    sns.kdeplot(numeric_df[col].values, bw=0.5,label='Train')
    # sns.kdeplot(raw_test[col].values, bw=0.5,label='Test')
   
plt.show() 


# Note: I highly recommend using Orange miner as a way to look at your data faster, then coming over to python and making prettier plots here :)
# 
# Let's start by comparing some categorical features from CoinGecko to the numeric Liquidity metric from CMC. Because Coin Gecko and Coin Market Cap both use differnt equations for formulating their respective metrics, it will be interesting to see if there are any statistically significant groupings.
# 
# Does acces to a websocket affect liquidity by CMC's standards?

# In[ ]:


# Set styleee
sns.set_style("dark")

# Get boxplot
plot = sns.violinplot(exchange_df['Websocket'], exchange_df['CMC Liquidity'])
plot = sns.stripplot(x='Websocket', y='CMC Liquidity', data=exchange_df, color="purple", jitter=0.3, size=3.0)
# Get statistical info
add_stat_annotation(plot, data=exchange_df, x=exchange_df['Websocket'], y=exchange_df['CMC Liquidity'],
                                   box_pairs=[("Available", "Not")],
                                   test='t-test_ind', text_format='star',
                                   loc='outside', verbose=2)


# How about API trading? in general?

# In[ ]:


plot = sns.violinplot(exchange_df['Trading_via_API'], exchange_df['CMC Liquidity'])
plot = sns.stripplot(x='Trading_via_API', y='CMC Liquidity', data=exchange_df, color="purple", jitter=0.3, size=3.0)
add_stat_annotation(plot, data=exchange_df, x=exchange_df['Trading_via_API'], y=exchange_df['CMC Liquidity'],
                                   box_pairs=[("Available", "Not")],
                                   test='t-test_ind', text_format='star',
                                   loc='outside', verbose=2)


# How about Websockets for CG's Lqicuidity? Not so much

# In[ ]:


plot = sns.violinplot(exchange_df['Websocket'], exchange_df['Liquidity'])
plot = sns.stripplot(x='Websocket', y='Liquidity', data=exchange_df, color="purple", jitter=0.3, size=3.0)
add_stat_annotation(plot, data=exchange_df, x=exchange_df['Websocket'], y=exchange_df['Liquidity'],
                                   box_pairs=[("Available", "Not")],
                                   test='t-test_ind', text_format='star',
                                   loc='outside', verbose=2)


# CMC Liquidity also seems to be affected by risk of sanctioning, where a low risk leads to higher liquidity

# In[ ]:


order = [" Low", " Medium", " High"]
plot = sns.boxplot(exchange_df['Sanctions'], exchange_df['CMC Liquidity'], order = order)
plot = sns.stripplot(x='Sanctions', y='CMC Liquidity', data=exchange_df, color="purple", jitter=0.3, size=3.0 ,order = order)
add_stat_annotation(plot, data=exchange_df, x=exchange_df['Sanctions'], y=exchange_df['CMC Liquidity'], order = order,
                                   box_pairs=[(" Low", " High"), (" Low", " Medium")],
                                   test='Mann-Whitney', text_format='star',
                                   loc='outside', verbose=2)


# Does negative news affect liquidity? We would expect high amounts of negative news to lead to lower liquidity, but quite the opposite is true. From a statistical perspective, here are already very few points in the "High" bin, and they are quite spread out, making this a sparse feature. Additionally, from an intuitive stand point, larger exchanges probably generate more news in general, so they will have already be generating a lot of liquidity as it is.
# 
# Either way you split it, there's no statistical difference between the different bins, so we can conclude negative news has a non-existent effect on liquidity.

# In[ ]:


order = [' High', ' Medium', ' Low']
plot = sns.boxplot(exchange_df['Negative_News'], exchange_df['CMC Liquidity'], order = order)
plot = sns.stripplot(x='Negative_News', y='CMC Liquidity', data=exchange_df, color="purple", jitter=0.3, size=3.0 ,order = order)
add_stat_annotation(plot, data=exchange_df, x=exchange_df['Negative_News'], y=exchange_df['CMC Liquidity'], order = order,
                                   box_pairs=[(" Low", " High"), (" Low", " Medium"), (' High', ' Medium')],
                                   test='Mann-Whitney', text_format='star',
                                   loc='outside', verbose=1)


# Earlier we made the new column "Bad_Pairs", so let's make some plots of that now to get an idea..

# In[ ]:


plot = sns.scatterplot(data = exchange_df, x = 'Bad_Pairs', y = 'CMC Liquidity', hue = 'Trust_Score', x_jitter=2.0)


# Hmm, maybe this was a bad idea to bin the columns? Let's just try red pairs now

# In[ ]:


plot = sns.scatterplot(data = exchange_df, x = 'Red_Pairs', y = 'CMC Liquidity', hue = 'Trust_Score', x_jitter=2.0)


# We saw from the heatmap previously that Bad_Pairs and Average BidAsk Spread are negatively correlated. Why is this the case? I would have expected opposite behavior

# In[ ]:


plot = sns.scatterplot(data = exchange_df, x = 'Bad_Pairs', y = 'BidAsk Spread', hue = 'Trust_Score', x_jitter=2.0)


# Looking at the graph, there isn't much spread in the data. Values are locked at the x and y-limits, which makes me doubt the relationship. Even at high levels of bad pairs, most of them sit in the 1.0 bid-ask spread range.
# 
# \\FUTURE\\ Look into getting CG's data as timeseries, then calculate the averages for the whole dataset (don't inner join CMC)

# We also noticed a moderate correlation between CG scale and CMC liquidity. Let's plot it.

# In[ ]:


def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

# sns.jointplot(data = exchange_df, x = 'Scale', y = 'CMC Liquidity', kind = 'reg', stat_func=r2)
sns.regplot(data = exchange_df, x = 'Scale', y = 'CMC Liquidity')


# But now lets get to the heart of what we really came to see, what is the relationship between CMC's liquidity and CG's trust score?

# In[ ]:


plot = sns.jointplot(data = exchange_df, x = 'Trust_Score', y = 'CMC Liquidity', kind = 'reg', stat_func=r2)


# In[ ]:




