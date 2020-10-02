#!/usr/bin/env python
# coding: utf-8

# Hello guys. Today I just finished my kernel on the visualization of modern portfolio theory. My kernel will briefly discuss and visualize the modern portfolio theory, in which a large number of brokers and portfolio managers are adapting. If you feel my kernel is helpful, please give me an upvote for further motivation.

# First of all, let's import our dataset and other useful libs

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import gmean
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.


# In[ ]:


price_df = pd.read_csv('../input/nyse/prices.csv')
sec_df = pd.read_csv('../input/nyse/securities.csv')
fund_df = pd.read_csv('../input/nyse/fundamentals.csv')


# In[ ]:


plt.figure(figsize=(15, 6))
ax = sns.countplot(y='GICS Sector', data=sec_df)
plt.xticks(rotation=45)


# In[ ]:


price_df.head()


# In[ ]:


price_df.isna().sum()


# We have the securities dataset and the stock prices dataset. What we want to do next is to fill in the stock prices dataset the sector of each stocks. In order to do this, we are gonnna use the pd.merge function:

# In[ ]:


sec_df = sec_df.rename(columns = {'Ticker symbol' : 'symbol','GICS Sector' : 'sector'})
sec_df.head()


# In[ ]:


price_df  = price_df.merge(sec_df[['symbol','sector']], on = 'symbol')
price_df['date'] = pd.to_datetime(price_df['date'])
price_df.head()


# For simplicity and relevant datas, we will only analyze the stock prices from the year of 2016 and above

# In[ ]:


price_df = price_df[price_df['date'] >= '2016-01-01']


# **1. Correlation Matrix**

# In[ ]:


sector_pivot = pd.pivot_table(price_df, values = 'close', index = ['date'],columns = ['sector']).reset_index()
sector_pivot


# In[ ]:


plt.figure(figsize = (10,10))
sns.heatmap(sector_pivot.corr(),annot=True, cmap="coolwarm")


# We have plotted the correlation matrix of our Sectors. The higher the correlation, the more likely that the stocks are moving in the same direction. To put it simply, if two stocks are highly correlated, they are likely to increase or decrease in price together. 
# 
# When building a diversified portfolio, investors seek negatively correlated stocks. Doing so reduces the risk of catastrophic losses in the portfolio and helps the investor sleep better at night. Assume the portfolio consists of two stocks and they are negatively correlated. This implies that when the price of one performs worse than usual, the other will likely do better than usual. However, risk takers would love to seek for positively correlated stocks for higher expected return, and of course, higher risk.

# In[ ]:


price_df['return'] = np.log(price_df.close / price_df.close.shift(1)) + 1
price_df['good'] = price_df['symbol'] == price_df['symbol'].shift(1)
price_df = price_df.drop(price_df[price_df['good'] == False].index)
price_df.dropna(inplace = True)


# **2. Portfolio selection by sector**

# In[ ]:


risk_free = 0.032
sector_df = pd.DataFrame({'return' : (price_df.groupby('sector')['return'].mean() - 1) * 252, 'stdev' : price_df.groupby('sector')['return'].std()})
sector_df['sharpe'] = (sector_df['return'] - risk_free) / sector_df['stdev']
plt.figure(figsize = (12,8))
ax = sns.barplot(x= sector_df['sharpe'], y = sector_df.index)


# Sharpe ratio is often used to describe how good our portfolio is. The higher the sharpe ratio, the better the portfolio. A sharpe ratio more than 1 is acceptable to investors. As a matter of fact, we will only choose the sectors that have sharpe ratio more than 1.

# In[ ]:


port_list = sector_df[sector_df['sharpe'] >= 1].index
port_list


# In[ ]:


price_df.head()


# After having the list of sectors, we will choose from each sectors the most outstanding return stock. In real life you should choose many. However, in this example, I will pick only one for simple illustration.

# In[ ]:


port_stock = []
return_stock = []
def get_stock(sector):
    list_stocks = price_df[price_df['sector'] == sector]['symbol'].unique()
    performance = price_df.groupby('symbol')['return'].apply(lambda x : (gmean(x) - 1) * 252).sort_values(ascending = False)
    
    for i in range(len(performance)):
        if performance.index[i] in list_stocks:
            port_stock.append(performance.index[i])
            return_stock.append(performance[i])
            break
    
for sector in port_list:
    get_stock(sector)

return_stock


# In[ ]:


port_df = price_df[price_df['symbol'].isin(port_stock)].pivot('date','symbol','return')


# **3. Porfolio risk and return calculation**

# Each portfolio will have different stock weights, or the allocation of your investment into each stock.
# ![](https://i.stack.imgur.com/IrPam.png)
# 
# The portfolio variance can be calculated by the formula above. In another word, to calculate the variance, we take the covariance matrix multiply with the weighted matrix and the tranposed of the weighted matrix
# ![](https://i.stack.imgur.com/o6aIy.png)

# In[ ]:


return_pred = []
weight_pred = []
std_pred = []
for i in range(1000):
    random_matrix = np.array(np.random.dirichlet(np.ones(len(port_stock)),size=1)[0])
    port_std = np.sqrt(np.dot(random_matrix.T, np.dot(port_df.cov(),random_matrix))) * np.sqrt(252)
    port_return = np.dot(return_stock, random_matrix)
    return_pred.append(port_return)
    std_pred.append(port_std)
    weight_pred.append(random_matrix)


# In[ ]:


pred_output = pd.DataFrame({'weight' : weight_pred , 'return' : return_pred, 'stdev' :std_pred })
pred_output['sharpe'] = (pred_output['return'] - risk_free) / pred_output['stdev']
pred_output.head()


# In[ ]:


max_pos = pred_output.iloc[pred_output.sharpe.idxmax(),:]
safe_pos = pred_output.iloc[pred_output.stdev.idxmin(),:]


# After running 2000 simulations, we finally plot the results, as well as the options for the portfolio, either the best performing or the safest one for risk adverse. 

# In[ ]:


plt.subplots(figsize=(15,10))
#ax = sns.scatterplot(x="Stdev", y="Return", data=pred_output, hue = 'Sharpe', size = 'Sharpe', sizes=(20, 200))

plt.scatter(pred_output.stdev,pred_output['return'],c=pred_output.sharpe,cmap='OrRd')
plt.colorbar()
plt.xlabel('Volatility')
plt.ylabel('Return')

plt.scatter(max_pos.stdev,max_pos['return'],marker='^',color='r',s=500)
plt.scatter(safe_pos.stdev,safe_pos['return'],marker='<',color='g',s=500)
#ax.plot()


# In[ ]:


print("The highest sharpe porfolio is {} sharpe, at {} volitality".format(max_pos.sharpe.round(3),max_pos.stdev.round(3)))

for i in range(len(port_stock)):
    print("{} : {}%".format(port_stock[i],(max_pos.weight[i] * 100).round(3)))


# In[ ]:


print("The safest porfolio is {} risk, {} sharpe".format(safe_pos.stdev.round(3), safe_pos.sharpe.round(3)))
for i in range(len(port_stock)):
    print("{} : {}%".format(port_stock[i],(safe_pos.weight[i] * 100).round(3)))


# Thank you for your attention. I hope you have a glance through the modern portfolio theory after seeing my kernel. Please give me an upvote for further motivation for finance - related projects. Thanks a lot!
