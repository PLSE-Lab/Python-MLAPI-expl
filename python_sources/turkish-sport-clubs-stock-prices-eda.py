#!/usr/bin/env python
# coding: utf-8

# # The Beginning
# We are going to explore datas of stock prices from same sectors. I love sports so I decided to analyze stocks of Galatasaray, Fenerbahce and Besiktas which are three biggest football clubs in Turkey. Let's begin!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# This is the dataset of daily OHLCV prices of every stock in XU100
# 
# Feel free to use it.

# **Import Python Modules **

# In[ ]:


import pandas as pd
import numpy as np 
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from datetime import datetime
#from __future__ import division


# **Importing the Stock Prices**

# In[ ]:


FB=pd.read_csv('../input/FENER_MG.csv',names=["Stock","DateTime","Open","High","Low","Close","Volume","0"])
GS=pd.read_csv('../input/GSRAY_MG.csv',names=["Stock","DateTime","Open","High","Low","Close","Volume","0"])
BJK=pd.read_csv('../input/BJKAS_MG.csv',names=["Stock","DateTime","Open","High","Low","Close","Volume","0"])


# In[ ]:


GS=GS.drop("0",axis=1)
BJK=BJK.drop("0",axis=1)
FB=FB.drop("0",axis=1)


# In[ ]:


FB['DateTime']=pd.to_datetime(FB['DateTime'])
GS['DateTime']=pd.to_datetime(GS['DateTime'])
BJK['DateTime']=pd.to_datetime(BJK['DateTime'])

GS['Stock']=GS['Stock'].astype(str).str[6:]
FB['Stock']=FB['Stock'].astype(str).str[6:]
BJK['Stock']=BJK['Stock'].astype(str).str[6:]


# In[ ]:


FB.index=FB.DateTime
GS.index=GS.DateTime
BJK.index=BJK.DateTime

FB=FB.drop("DateTime",axis=1)
GS=GS.drop("DateTime",axis=1)
BJK=BJK.drop("DateTime",axis=1)


# In[ ]:


BJK.head()


# In[ ]:


FB.head()


# In[ ]:


GS.head()


# When we look at dates of datas we can see that FB's start date is 2 years later than other clubs. For proper analysis, we need all datas start from the same date.  

# In[ ]:


GS=GS["2004-02-20":]
BJK=BJK["2004-02-20":]


# Now we are even!

# Now we will look at the graphs of **close prices of GS, FB and BJK **

# In[ ]:


plt.plot(GS["Close"])


# In[ ]:


plt.plot(FB["Close"])


# In[ ]:


plt.plot(BJK["Close"])


# # Some Comments about Close Prices of Teams
# 
# The first thing that attract my attention is the graphs of FB and GS are resemble each other with slight differences until 2013 and volatility of BJK is higher than other assets. In my opinion, the reason of rapid decrease of prices in 2011 is chicane lawsuit that affects all sports clubs in Turkey and it caused a panic in investments in sports clubs, especially from foreigner investors. Moreover, the reason of the jump in 2016 of the stock price of BJK was they were really succesful both in Champions League and Turkish Super League, they were leader in their Champions League group and their game was near perfect in Turkish Super League. Surely there are other factors that affect stock prices of these assets like country risk and the strategies that these clubs CEO's follow. However, I find these jumps and downs more interesting.

# We have stock price for 15 years starting from 2004 to 2019 

# Volume traded for BJK Stock

# In[ ]:


plt.plot(BJK["Volume"])


# As I mentioned right before, there is a huge increase in volume of BJK stock traded in the season 2016-2017. When people heard some speculations from somewhere like friends, commentors or financial analysts, they tend to invest easily. It is the most basic reason why stocks got overvalued so frequently. That's why BJK's stock price value and trading volume increased rapidly, actually they are both peak points, and then they both returned to normalcy. To conclude, we can see the effect of success in football of BJK in both graphs, volume and price. 

# **Daily Returns**

# In[ ]:


FB['Daily Return']=FB['Close'].pct_change()

FB['Daily Return'].plot(figsize=(15,4),legend=True,linestyle='--',marker='o')
plt.ioff()


# In[ ]:


GS['Daily Return']=GS['Close'].pct_change()
GS['Daily Return'].plot(figsize=(15,4),legend=True,linestyle='--',marker='o')
plt.ioff()


# What a fragile sector to invest, ha!

# **Average Daily return**

# In[ ]:


FB['Daily Return'].hist(bins=50)
plt.ioff()


# So the stock fluctuation follows a right skewed normal distribution between +20% and -20% for FB !!!

# # Correlations
# 
# **Checking if the stock prices of sports clubs are correlated**
# 
# Before calculating correlations, if you are unfamiliar with Turkish Super League, there is something you should know. Galatasaray, Fenerbahce and Besiktas are the biggest three football and sport clubs in Turkey. However, Galatasaray has 21, Fenerbahce has 19 and Besiktas has just 15 championships. The derbies between Galatasaray and Fenerbahce is always more intense than other derbies. So we can count GS and FB are best two teams in the league and Besiktas comes after them. So we expect that these two teams' stock prices are more correlated.

# In[ ]:


df=FB.index.copy()
df=pd.DataFrame(df)
df.index=df.DateTime
df['FB']=FB['Close']
df['GS']=GS['Close']
df['BJK']=BJK['Close']


# In[ ]:


df=df.drop("DateTime",axis=1)


# In[ ]:


sport_rets=df.pct_change()
sport_rets=pd.DataFrame(sport_rets)
sport_rets.shape


# **Compare Galatasaray & Fenerbahce first**

# In[ ]:


from scipy import stats
sns.jointplot('GS','FB',sport_rets,kind='scatter',color='seagreen').annotate(stats.pearsonr)
plt.ioff()


# * We can see that p =0.31

# In[ ]:


from scipy import stats
sns.jointplot('FB','BJK',sport_rets,kind='scatter',color='seagreen').annotate(stats.pearsonr)
plt.ioff()


# * We can see that p =0.17

# In[ ]:


from scipy import stats
sns.jointplot('GS','BJK',sport_rets,kind='scatter',color='seagreen').annotate(stats.pearsonr)
plt.ioff()


# * We can see that p =0.15
# 
# As we expected, the correlation between FB and GS significantly higher than other correlations. When we look at p-values, we can't reject any correlation so we can say that all assets are correlated with each other.

# **Heatmap Daily Return**

# In[ ]:


sns.heatmap(sport_rets.corr(),annot=True,cmap='summer',linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)
plt.ioff()


# For a more clear demonstration of correlations, we can look at the above table. All companies have positive correlations. In my opinion, the reason is they are in the same country and from same sector and their prices are affected from similar news and announcements 

# **Risk Analysis**

# In[ ]:


rets=sport_rets.dropna()


# In[ ]:


area=np.pi*5
plt.scatter(rets.mean(),rets.std(),s=area)
plt.xlabel('Expected Return')
plt.ylabel('Risk')

for label, x,y in zip(rets.columns,rets.mean(),rets.std()):
    plt.annotate(
        label,
        xy=(x,y),xytext=(22,-3),
        textcoords='offset points',ha='right',va='bottom',
        arrowprops=dict(arrowstyle='-',connectionstyle='arc,rad=-0.3'))

for label,x,y in zip(rets.columns,rets.mean(),rets.std()):
    print(label + " Expected Return: " + str(round(x,5)) + " - Standard Deviation: " + str(round(y,3)))


# Keep in mind that we are looking for lower risk and higher expected return. As expected, when the risk increases expected return increases as well.
# 
# Don't forget to tell me which team you would invest, if you were deciding which asset to invest only looking at historical data in **comments**.

# In[ ]:


print("If you invest in BJK, 95% of the times the worst daily loss will not exceed " + str(np.round(rets['BJK'].quantile(0.05),3)) + "%")
print("If you invest in GS, 95% of the times the worst daily loss will not exceed " + str(np.round(rets['GS'].quantile(0.05),3))+ "%")
print("If you invest in FB, 95% of the times the worst daily loss will not exceed " + str(np.round(rets['FB'].quantile(0.05),3))+ "%")


# # Thanks for reading!
# 
# I hope you enjoyed while reading as much as I enjoyed while I was writing. 
# 
# I decided to continue this notebook. And Keep Steady!! I will search for the correlation between the results of important matches of these teams and change of their stock prices. I hope we will find something that could be benefitial for investing.
# 
# Waiting for your questions, critisims and advices. 
# 
# Best. 
# Ege
