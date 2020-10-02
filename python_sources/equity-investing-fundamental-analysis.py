#!/usr/bin/env python
# coding: utf-8

# # Equity Investing -  Fundamental Analysis
# 
# Filter stocks based on fundamentals

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Get-fundamental-data" data-toc-modified-id="Get-fundamental-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Get fundamental data</a></span></li><li><span><a href="#Nifty-500-sector-breakdown" data-toc-modified-id="Nifty-500-sector-breakdown-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Nifty 500 sector breakdown</a></span></li><li><span><a href="#Filter-based-on-fundamentals" data-toc-modified-id="Filter-based-on-fundamentals-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Filter based on fundamentals</a></span></li><li><span><a href="#Visualize-selected-stocks" data-toc-modified-id="Visualize-selected-stocks-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Visualize selected stocks</a></span></li></ul></div>

# In[ ]:


import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.pyplot import figure

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.set_context('talk')
matplotlib.rcParams['font.family'] = 'arial'


# ## Get fundamental data

# In[ ]:


df = pd.read_csv("../input/nifty_500_stats.csv", 
                              sep=';', 
                              encoding='utf-8', 
                              thousands=',',
                              na_values='None'
                             )
del df['Unnamed: 0']
df['price_bookvalue'] = np.round(df.current_value / df.book_value, 2)
df['peg3'] = df.price_earnings / df.sales_growth_3yr

# display(df.info())
df.sample(5).T


# In[ ]:


df = df.fillna(0)


# In[ ]:


ind_ratio = pd.DataFrame(df.groupby('industry')['price_earnings'].quantile(.8)).reset_index()
ind_ratio = ind_ratio.rename(columns = {'price_earnings': 'ind_pe80'})
df = pd.merge(df, ind_ratio,  how='left', left_on=['industry'], right_on = ['industry'])

# Above avg ROCE and ROE
ind_ratio = pd.DataFrame(df.groupby('industry')['roce', 'roe'].mean()).reset_index()
ind_ratio = ind_ratio.rename(columns = {'roce': 'ind_roce50',
                                       'roe': 'ind_roe50'})
df = pd.merge(df, ind_ratio,  how='left', left_on=['industry'], right_on = ['industry'])


# ## Nifty 500 sector breakdown

# In[ ]:


f = {'market_cap':['sum'], 'company':['count']}

industries = df.groupby('industry').agg(f)
industries.columns = industries.columns.get_level_values(0)
industries = industries.reset_index()
industries['company'] = 100*industries['company'] / industries['company'].sum()
industries['market_cap'] = 100*industries['market_cap'] / industries['market_cap'].sum()
industries = industries.sort_values('market_cap', ascending=False)

fig = figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(1, 2, 1)
ax1 = sns.barplot(x="company", y="industry", data=industries, palette=("Greys_d"))
ax1.set_xlabel('Percentage of Companies', weight='bold')
ax1.set_ylabel('Industry', weight = 'bold')
ax1.set_title('Nifty 500 sector breakdown\n')

plt.subplot(1, 2, 2)
ax2 = sns.barplot(x="market_cap", y="industry", data=industries, palette=("Greens_d"))
ax2.set_xlabel('Market Capitalization', weight='bold')
ax2.set_ylabel('')
ax2.set_yticks([])

sns.despine()
plt.tight_layout();


# In[ ]:


f = {'price_bookvalue':['mean'], 'price_earnings':['mean'], 'peg3':['mean']}

ratios = df.groupby('industry').agg(f)
ratios.columns = ratios.columns.get_level_values(0)
ratios = ratios.reset_index()
ratios = ratios.sort_values('price_earnings', ascending=False)

fig = figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(1, 3, 1)
ax1 = sns.barplot(x="price_bookvalue", y="industry", data=ratios, palette=("Greys_d"))
ax1.set_xlabel('P/B', weight='bold')
ax1.set_ylabel('Industry', weight = 'bold')
ax1.set_title('Average across industries\n')

plt.subplot(1, 3, 2)
ax2 = sns.barplot(x="price_earnings", y="industry", data=ratios, palette=("Greens_d"))
ax2.set_xlabel('P/E', weight='bold')
ax2.set_ylabel('')
ax2.set_yticks([])

plt.subplot(1, 3, 3)
ax2 = sns.barplot(x="peg3", y="industry", data=ratios, palette=("Greens_d"))
ax2.set_xlabel('PEG3', weight='bold')
ax2.set_ylabel('')
ax2.set_yticks([])

sns.despine()
plt.tight_layout();


# ## Filter based on fundamentals

# In[ ]:


df_1 = df.copy()
print('Total:', df_1.shape[0])
print('\n')

df_filter = df_1[(df_1.price_bookvalue > 0)]
removed = list(set(df_1.symbol) - set(df_filter.symbol))

print('P/B > 0 :', df_filter.shape[0])
print('Removed: ', len(removed))
print(removed)
print('\n')

# P/E > 0%
df_1 = df_filter.copy()
df_filter = df_1[(df_1.price_earnings > 0)]
removed = list(set(df_1.symbol) - set(df_filter.symbol))

print('P/E > 0 :', df_filter.shape[0])
print('Removed: ', len(removed))
print(removed)
print('\n')

df_1 = df_filter.copy()
df_filter = df_1[(df_1.peg3 > 0)]
removed = list(set(df_1.symbol) - set(df_filter.symbol))

print('P/E to Sales growth in 3yrs > 0 :', df_filter.shape[0])
print('Removed: ', len(removed))
print(removed)
print('\n')

df_1 = df_filter.copy()
df_filter = df_1[(df_1.price_bookvalue <= df.price_bookvalue.quantile(.8))]
removed = list(set(df_1.symbol) - set(df_filter.symbol))

print('P/B <= 80 precentile of Nifty 500 :', df_filter.shape[0])
print('Removed: ', len(removed))
print(removed)
print('\n')

# P/E <= 80 percentile of Industry
df_1 = df_filter.copy()
df_filter = df_1[(df_1.price_earnings <= df_1.ind_pe80)]
removed = list(set(df_1.symbol) - set(df_filter.symbol))

print('P/E <= 80 precentile of Industry :', df_filter.shape[0])
print('Removed: ', len(removed))
print(removed)
print('\n')

# Above avg ROCE
df_1 = df_filter.copy()
df_filter = df_1[(df_1.roce > df_1.ind_roce50)]
removed = list(set(df_1.symbol) - set(df_filter.symbol))

print("ROCE above Industry's average :", df_filter.shape[0])
print('Removed: ', len(removed))
print(removed)
print('\n')

# Above avg ROE
df_1 = df_filter.copy()
df_filter = df_1[(df_1.roe > df_1.ind_roe50)]
removed = list(set(df_1.symbol) - set(df_filter.symbol))

print("ROE above Industry's average :", df_filter.shape[0])
print('Removed: ', len(removed))
print(removed)
print('\n')

df_1 = df_filter.copy()


# In[ ]:


del df_1['ind_pe80']
del df_1['ind_roce50']
del df_1['ind_roe50']


# ## Visualize selected stocks

# In[ ]:


from scipy.stats import norm

fig = figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(2, 3, 1)
ax1 = sns.distplot(df_1.price_bookvalue, rug=True, kde=True) 

plt.subplot(2, 3, 2)
ax2 = sns.distplot(df_1.price_earnings, rug=True, kde=True) 

plt.subplot(2, 3, 3)
ax3 = sns.distplot(df_1.roce, rug=True, kde=True) 

plt.subplot(2, 3, 4)
ax4 = sns.distplot(df_1.roe, rug=True, kde=True) 

plt.subplot(2, 3, 5)
ax5 = sns.distplot(df_1.dividend_yield, rug=True, kde=True) 

plt.subplot(2, 3, 6)
ax6 = sns.distplot(df_1.sales_growth_3yr, rug=True, kde=True) 

sns.despine()
plt.tight_layout();


# In[ ]:


print("Number of stocks by market capitalization:")
display(df_1.groupby('category')['company'].size())

# Nifty 50 
print("Nifty 50:")
display(df_1[(df_1.category == 'Nifty 50')].symbol.values)

# Nifty Next 50
print("Nifty Next 50:")
display(df_1[(df_1.category == 'Nifty Next 50')].symbol.values)

# Nifty Midcap 150 
print("Nifty Midcap 150:")
display(df_1[(df_1.category == 'Nifty Midcap 150')].symbol.values)

# Nifty Smallcap 250 
print("Nifty Smallcap 250:")
display(df_1[(df_1.category == 'Nifty Smallcap 250')].symbol.values)


# In[ ]:




