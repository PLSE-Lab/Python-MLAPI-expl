#!/usr/bin/env python
# coding: utf-8

# # How well is Turkey doing compared to the European Union?
# 
# - This analysis starts with a analysis of the level of trade and it's comparaison to the urbanisation of Turkey and the European Union.
# - Secondly, the three main aspects of the Human Development Index are taken into account seperatly (Health, Education and Income) for this three aspects Life Expactancy at Birth, Enrolment in secondary education and GDP per capita, PPP are used.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
plt.style.use('ggplot')


# In[ ]:


df = pd.read_csv('/kaggle/input/world-development-indicators/Indicators.csv')

Indicator_array =  df[['IndicatorName','IndicatorCode']].drop_duplicates().values


# In[ ]:


import pandas as pd
Country = pd.read_csv("../input/world-development-indicators/Country.csv")
CountryNotes = pd.read_csv("../input/world-development-indicators/CountryNotes.csv")
Footnotes = pd.read_csv("../input/world-development-indicators/Footnotes.csv")
Indicators = pd.read_csv("../input/world-development-indicators/Indicators.csv")
Series = pd.read_csv("../input/world-development-indicators/Series.csv")
SeriesNotes = pd.read_csv("../input/world-development-indicators/SeriesNotes.csv")


# In[ ]:


chosen_country1 = 'Turkey'
chosen_indicators = ['NE.TRD.GNFS.ZS', 'SE.SEC.NENR','SP.DYN.LE00.IN',                      'NY.GDP.PCAP.PP.KD','SP.URB.TOTL.IN.ZS']

df_subset = df[df['IndicatorCode'].isin(chosen_indicators)]


df_EU = df_subset[df['CountryName']=="European Union"]
df_Turkey = df_subset[df['CountryName']=="Turkey"]


# In[ ]:


def plot_indicator(indicator,delta=10):
    ds_EU = df_EU[['IndicatorName','Year','Value']][df_EU['IndicatorCode']==indicator]
    try:
        title = ds_EU['IndicatorName'].iloc[0]
    except:
        title = "None"

    xeu = ds_EU['Year'].values
    yeu = ds_EU['Value'].values
    ds_Turkey = df_Turkey[['IndicatorName','Year','Value']][df_Turkey['IndicatorCode']==indicator]
    xturkey = ds_Turkey['Year'].values
    yturkey = ds_Turkey['Value'].values
    
    plt.figure(figsize=(14,4))
    
    plt.subplot(121)
    plt.plot(xeu,yeu,label='European Union')
    plt.plot(xturkey,yturkey,label='Turkey')
    plt.title(title)
    plt.legend(loc=2)


# ### Trade as percentage of GDP
# 
# - We see a rising trend in both the EU and Turkey but even thought we observe a lowering in the gap between them it's hard to say that Turkey is close to catch up to the European Union average

# In[ ]:


plot_indicator(chosen_indicators[0],delta=10)


# ### Percent of the total population living in urban areas
# 
# - This gap (stated above) can be explained by the fact that over the years the urbanization in the European Union was significantly more then it was in Turkey which set the base for them to have a higher trade rate in their GDP
# - We see that Turkey has a very steep slope and is catching up to the European Union in terms of urbanization but historical persistence explains why the gap between the trade percentage of GDP between these two regions are not shrinking in the same percentages as in the rise in the urbanization
# - Overall we can say that the rise in urbanization is correlated to the rising percentage of trade contribution to GDP

# In[ ]:


plot_indicator(chosen_indicators[4],delta=10)


# ### Enrolment in secondary education
# 
# - Once more we see the importance of historical persistence and early [neolithic transition](http://https://en.wikipedia.org/wiki/Neolithic_Revolution) of the more western countries
# - The gap between the secondary education enrolment rates in the European Union and Turkey used to be very high, mainly because of the early-adopter advantages the Europe enjoyed which can be explained by a unified set of countries which reinforces development whereas Turkey's overall instability and high levels of rural population led them to have lower levels of education

# In[ ]:


plot_indicator(chosen_indicators[1],delta=10)


# ### Life expectancy at birth
# 
# - When we look at the starting point of the graph we see a very interesting point where in Turkey in 1960 one would expect to live only 45 years and the same rate in the European Union average is 70, a number Turkey only reached around the year 2000
# - Althought Turkey has an overall higher growing rate in their life expectancy we see a stabilisation of the rate
# - Even if Turkey does not have a life expectancy as high as the European Union average they're well above the world average of 71.1 years
# - Overall Turkey is not in a negative absolute state but in a comparative negative state when we take the European Union as a proxy
# 

# In[ ]:


plot_indicator(chosen_indicators[2],delta=10)


# ### GDP per capita, Purchasing Power Parity
# 
# - This metric is one of the most critical ones and also one where we observe the most difference between the European Union and Turkey
# - We measure the purchasing power parity in order to have a more realistic view on the matter and we peg the currency to the 2011 dollar
# - Turkey is far below the level the European Union was in the 1990, thus we see a huge difference in the purchasing power enjoyed by these two regions
# - Turkey is (statistically) not expected to catch up or even come close to the European Union in this very crucial metric
# - The purchasing power is direclty correlated with living standarts and this indirectly boosts life satisfaction (in most cases, exception are emprically visble see; [Easterlin Paradox](http://https://en.wikipedia.org/wiki/Easterlin_paradox))

# In[ ]:


plot_indicator(chosen_indicators[3],delta=10)


# ### Lorenz Curve and Gini Coeefficient (2016)
# 
# - The blue line in the curve reflects the perfect equality case where the bottom 20 percent holds 20 percent of the income of the nation and so on, which is obviously not statistically realistic and relevant
# - The red line is the actual distribution of wealth in Turkey, for example the top 20 percent holds 48.3 percent of the total income in the country country whereas in a perfect world they should have 20 percent
# - We compute a gini coeficient in order to measure the severity of the income distribution inequality in the country which is the area between the red and blue line divided by 0.5
# - We get a number between 0 and 1 which is a measure of how uneaqually wealth is distributed in the country
# - Turkey has a gini cooeficient of 0.39 which is above the European Union average of [0.30](http://https://www.oecd.org/els/soc/cope-divide-europe-2017-background-report.pdf)

# In[ ]:


arr = np.array([5.7,9.9,14.5,21.6,48.3])

def gini(arr):
    count = arr.size
    coefficient = 2 / count
    indexes = np.arange(1, count + 1)
    weighted_sum = (indexes * arr).sum()
    total = arr.sum()
    constant = (count + 1) / count
    return coefficient * weighted_sum / total - constant

def lorenz(arr):
    scaled_prefix_sum = arr.cumsum() / arr.sum()
    return np.insert(scaled_prefix_sum, 0, 0)

print(gini(arr))

lorenz_curve = lorenz(arr)

plt.plot(np.linspace(0.0, 1.0, lorenz_curve.size), lorenz_curve)

plt.plot([0,1], [0,1])
plt.show()

