#!/usr/bin/env python
# coding: utf-8

# # The Soybean Price

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pprint
df_train = pd.read_csv("../input/" + os.listdir("../input")[0])
df_train.Date = pd.to_datetime(df_train.Date)
df_train.set_index("Date", inplace=True)
# Any results you write to the current directory are saved as output.


# We have 48 columns and 20700 samples. Columns can be divided to few categorized part. First part is US regions where they produce soybean in significant amount temperature. We have high and low temperature of Soux Falls, Indianapollis, and Memphis. Second part is other common price of commodity such as gold, oil, and US index. Third part is pricec of soybean product derivative such as soybean, soymeal, and soyoil. Each product have their own OCHLV price. The fourth and last part is production area and capacity in countries view point. There are 11 area including US The rest column is date and open market price. 

# In[ ]:


us_regions = ["soux_H", 
              "soux_L", 
              "indianap_H", 
              "Indianap_L", 
              "memphis_H", 
              "memphis_L"]

commodity_price = ["Gold", "USD", "Oil"]

soy_OCHLV = ['Soy_Bean_high', 'Soy_Bean_low', 'Soy_Bean_settle',
       'Soy_Bean_volume', 'Soy_Bean_openint', 'Soy_Meal_high', 'Soy_Meal_low',
       'Soy_Meal_settle', 'Soy_Meal_volume', 'Soy_Meal_openint',
       'Soy_Oil_high', 'Soy_Oil_low', 'Soy_Oil_settle', 'Soy_Oil_volume',
       'Soy_Oil_openint']

production_origin = ['US_Area', 'US_Production', 'Brazil_Area',
       'Brazil_Production', 'Argentina_Area', 'Argentina_Production',
       'China_Area', 'China_Production', 'India_Area', 'India_Production',
       'Paraguay_Area', 'Paraguay_Production', 'Canada_Area',
       'Canada_Production', 'RussianF_Area', 'RussianF_Production',
       'CentAmer_Area', 'CentAmer_Production', 'Bolivia_Area',
       'Bolivia_Production', 'Africa_Area', 'Africa_Production']


# ## Column Conclusion
# 1. `Market_Open` is describing whether a market is open at that time. Usually market open only open for workdays. In special case such as holiday, market close although it is a workday. From this information, we can tell what day each sample is. 
# 
# 2. `Efficiency` can be defined as how much soybean produced per area allocated. Each source origin can have different efficiency based on agricultural technologies and policies. Efficency can be dropped because crop failure or farmer strike but again efficiency is heavily depend in agricultural technologies and policies. 
# 
# 3. All temperature is in Farenheit. 

# ### Temperature and Soybean Production

# In[ ]:


plt.figure(figsize=(20,6))
sns.distplot(df_train["soux_H"], label="SOUX FALL")
sns.distplot(df_train["indianap_H"], label="INDIANAPOLIS")
sns.distplot(df_train["memphis_H"], label= "MEMPHIS")
plt.title("HIGH TEMPERATURE DISTRIBUTION")
plt.xlabel("Temperature")
plt.ylabel("Probability")
plt.legend()
plt.grid()

plt.figure(figsize=(20,6))
sns.distplot(df_train["soux_L"], label="SOUX FALL")
sns.distplot(df_train["indianap_L"], label="INDIANAPOLIS")
sns.distplot(df_train["memphis_L"], label= "MEMPHIS")
plt.title("LOW TEMPERATURE DISTRIBUTION")
plt.xlabel("Temperature")
plt.ylabel("Probability")
plt.legend()
plt.grid()


# We have two graphics, first is the record of high temperature distribution and second is the record of low temperature. It's looks like Memphis is the hottest among three region. We need to find out about US soybean production and these temperatures. Let us cut the data so we only a year data, assuming that temperature have a one year period.

# In[ ]:


plt.figure(figsize=(20,6))
plt.title("HIGH TEMPERATURE PLOT IN ONE YEAR")
plt.plot(df_train['soux_H'].loc['1963-01-01':])
plt.plot(df_train['indianap_H'].loc['1963-01-01':])
plt.plot(df_train['memphis_H'].loc['1963-01-01':])
plt.legend(["SOUX FALL", "INDIANAPOLIS", "MEMPHIS"])
plt.grid()

plt.figure(figsize=(20,6))
plt.title("LOW TEMPERATURE PLOT IN ONE YEAR")
plt.plot(df_train['soux_L'].loc['1963-01-01':])
plt.plot(df_train['indianap_L'].loc['1963-01-01':])
plt.plot(df_train['memphis_L'].loc['1963-01-01':])
plt.legend(["SOUX FALL", "INDIANAPOLIS", "MEMPHIS"])
plt.grid()


# As we can see from plot above, Memphis have the highest among three areas and Soux Fall is the lowest among three areas. Let us take the middle value between high and low temperature, find the average middle temperature in three areas, and average it throughout the year. We will plot the year's temperature average to soybean productivity in each year. 

# In[ ]:


soux_mid_temp = ((df_train['soux_H'] - df_train['soux_L']) / 2) + df_train['soux_L']
indianap_mid_temp = ((df_train['indianap_H'] - df_train['indianap_L']) / 2) + df_train['indianap_L']
soux_mid_temp = ((df_train['memphis_H'] - df_train['memphis_L']) / 2) + df_train['memphis_L']

avg_temp = (soux_mid_temp + indianap_mid_temp + soux_mid_temp) / 3
yearly_temp_avg = [avg_temp.loc[str(i)].mean() for i in np.unique(df_train.index.year)]
US_soybean_prod = [df_train["US_Production"].loc[str(i)].mean() for i in np.unique(df_train.index.year)]

# deleting 1961 because the temperature only recorded in winter
del yearly_temp_avg[0]
del US_soybean_prod[0]

plt.figure(figsize=(14,14))
plt.title("TEMPERATURE AND SOYBEAN PRODUCTION")
plt.scatter(yearly_temp_avg, US_soybean_prod)
plt.plot(yearly_temp_avg, US_soybean_prod, alpha=0.3)
plt.xlabel("Temperature Averange")
plt.ylabel("Soybean Production")
plt.grid()

fig, ax1 = plt.subplots(figsize=(15,6))
ax1.set_title("TEMPERATURE AND SOYBEAN PRODUCTION")
ax1.plot(np.unique(df_train.index.year)[1:], yearly_temp_avg, '-r')
ax1.set_xlabel('Year')
ax1.set_ylabel('Temperature')
ax1.tick_params('y')
ax1.legend(["TEMPERATURE"], loc=1)

ax2 = ax1.twinx()
ax2.plot(np.unique(df_train.index.year)[1:], US_soybean_prod)
ax2.set_ylabel('Soybean Production')
ax2.tick_params('y')
ax2.legend(["PRODUCTION"], loc=0)
ax1.grid()


# We plot the soybean production and temperature in two ways, first we plot it in a scatter with a shadow line that represent time, second we plot it in line plot with different two y-axis where the left hand side is axis for temperature and the right hand side is axis for soybean production. In my opinion the second one is more readable than the first one.
# 
# We can see from the second graph that temperature and soybean production is rising but not necessarily in a linear way. In a few year, production can be high although the temperature is low and vice versa. The main problem finding relation whether temperature affect soybean production is we cannot isolate time. As we state earlier, soybean production is depend on advances on agriculture technologies which we assume always have a better technologies as time pass by. Maybe there is a correlation between temperature and soybean production but not much. 
# 
# From this two plot we can conclude inexpertly that soybean production capabilty become much more reliable as time progress.  We assume that crops does not grow well in cold temperature. In 2014, we have one of the lowest temperature yet have one of the highest temperature. Apparently high temperature fluctuation does not budge soybean production, at least in US. 

# In[ ]:


fig, ax1 = plt.subplots(figsize=(15,6))
ax1.set_title("TEMPERATURE AND SOYBEAN PRODUCTION")
ax1.plot(np.unique(df_train.index.year)[1:], yearly_temp_avg, '-r')
ax1.set_xlabel('Year')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Temperature')
ax1.tick_params('y')

ax2 = ax1.twinx()
ax2.plot(np.unique(df_train.index.year)[1:], US_soybean_prod)
ax2.set_ylabel('Soybean Production')
ax2.tick_params('y')
ax1.grid()


# ### Production Effieciency
# 
# We define production effieciecy as production divided by area. Theoritically, the most effiecient plantation is the one which can give more crops with limited land. We three dimension of data, the land which represented by country's area allocation for soybean plantation, time, production. Based on this, we can get effieciency of each land and how it is growth or decline over time. Below, we map production and area in each point of time in each land. Let us dig deeper.

# In[ ]:


plt.figure(figsize=(20,18))
average_production = []
for i in range(0,22,2):
    plt.scatter(np.log10(df_train[production_origin[i]]), np.log10(df_train[production_origin[i+1]]))
    average_production.append(df_train[production_origin[i+1]])
plt.grid()
plt.title("Number of Area and Production " + str(int(np.average(average_production))))
plt.xlabel("Area of Soybean Plantation - Log Scale")
plt.ylabel("Production of Soybean Plantation - Log Scale")
plt.legend(["US", "Brazil", "Argentina", "China", 
            "India", "Paraguay", "Canada", 
            "Russia", "Central America", "Bolivia", "Africa"])


# In[ ]:


for i in range(0, 22, 2):
    fig, ax1 = plt.subplots(figsize=(15,6))
    ax1.set_title(production_origin[i].split("_")[0] + " AREA AND SOYBEAN PRODUCTION")
    ax1.plot(df_train[production_origin[i]], '-r')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Area')
    ax1.tick_params('y')
    ax1.legend(["AREA"], loc=0)

    ax2 = ax1.twinx()
    ax2.plot(df_train[production_origin[i+1]])
    ax2.set_ylabel('Soybean Production')
    ax2.tick_params('y')
    ax2.legend(["PRODUCTION"], loc=4)
    ax1.grid()

plt.figure(figsize=(15,8))
for i in range(0,22,2):
    plt.plot(df_train[production_origin[i+1]]/df_train[production_origin[i]])
plt.ylabel("Efficiency")
plt.xlabel("Year")
plt.title("LAND EFFICENCY")
plt.legend(["US", "Brazil", "Argentina", "China", 
            "India", "Paraguay", "Canada", 
            "Russia", "Central America", "Bolivia", "Africa"])
plt.grid()


# India and Canada soybean production have dramatic leap in 2018 compared to the rest of countries. There are few interesting happen here. Central America and Africa lost many land for soybean production. Development in both areas are not as fast as US or India. This is strengthen our hypothesis about how agricultural technologies and policies really impace soybean production. Russia have many land dedicated for soybean production but have lower efficiency compared to another land. Overall soybean production is growing and many land opened for a new plantation except in Central America. Average soybean production in data we have is $11.513.517$. Let us calculate the average production and land overall.

# In[ ]:


fig, ax1 = plt.subplots(figsize=(15,6))
ax1.set_title("AVERAGE AREA AND SOYBEAN PRODUCTION")
ax1.plot(df_train[[production_origin[i] for i in range(0,22,2)]].mean(axis=1), '-r')
ax1.set_xlabel('Year')
ax1.set_ylabel('Area')
ax1.tick_params('y')
ax1.legend(["AREA"], loc=0)

ax2 = ax1.twinx()
ax2.plot(df_train[[production_origin[i+1] for i in range(0,22,2)]].mean(axis=1))
ax2.set_ylabel('Soybean Production')
ax2.tick_params('y')
ax2.legend(["PRODUCTION"], loc=4)
ax1.grid()

fig, ax1 = plt.subplots(figsize=(15,6))
ax1.set_title("SUM AREA AND SOYBEAN PRODUCTION")
ax1.plot(df_train[[production_origin[i] for i in range(0,22,2)]].sum(axis=1), '-r')
ax1.set_xlabel('Year')
ax1.set_ylabel('Area')
ax1.tick_params('y')
ax1.legend(["AREA"], loc=0)

ax2 = ax1.twinx()
ax2.plot(df_train[[production_origin[i+1] for i in range(0,22,2)]].sum(axis=1))
ax2.set_ylabel('Soybean Production')
ax2.tick_params('y')
ax2.legend(["PRODUCTION"], loc=4)
ax1.grid()


# ### Soybean Contribution
# We want to know how much a land contribute to whole world soybean production. The whole world means in avaliable data. So each year, we find the total production of soybean and compare it with each land soybean contribution. 

# In[ ]:


years = np.unique(df_train.index.year)
total_soybean = df_train[[production_origin[i+1] for i in range(0,22,2)]].sum(axis=1)

df_production = pd.DataFrame()
for i in range(0, 22, 2):
    df_production[production_origin[i+1]] = df_train[production_origin[i+1]] / total_soybean

proportion = pd.DataFrame([df_production.loc[str(i)].mean(axis=0) for i in years], index=years)
plt.figure(figsize=(20,40))
plt.title("PRPORTION OF SOYBEAN PRODUCTION")
sns.heatmap(proportion, fmt="f", annot=True, robust=True, cbar=False, linewidths=0.2, cmap="YlOrBr_r")


# It is like showing columns but with color. US dominating soybean production since the very beginning. Central America in mid 1900 also dominating soybean production but declining as time pass by. Brazil, slow but sure, become leading in soybean production after US. Although US dominate soybean production, its domination start declining as other land catch up with production in around early 90s. 

# ### Soybean Pricing and Other Commodities
# 
# In data we have, we have many pricing type especially in soybean and its derivative. In OCHLV, we only took the closing price for the day and we compare to other commodities such as gold, US index, and oil. 

# In[ ]:


plt.figure(figsize=(14,6))
plt.title("Commodities Price Overtime")
plt.plot(df_train['Gold'])
plt.plot(df_train['USD'])
plt.plot(df_train['Oil'])
plt.plot(df_train['bean_settle'])
plt.xlabel("Year")
plt.ylabel("Price")
plt.grid()
plt.legend(["Gold", "USD", "Oil", "SOYBEAN"])

def plot_long(Series, title):
    plt.figure(figsize=(14,3))
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Price")
    plt.grid()
    plt.plot(Series)
    
plot_long(df_train['Gold'].pct_change(), "Gold Price Change in Percent Overtime")
plot_long(df_train['USD'].pct_change(), "USD Price Change in Percent Overtime")
plot_long(df_train['Oil'].pct_change(), "Oil Price Change in Percent Overtime")
plot_long(df_train['bean_settle'].pct_change(), "Soybean Price Change in Percent Overtime")


# According to this plot, soybean is a volatile commodities compared to oil and USD. We also plot the percentage growth of each commodities. The plot we just made is to dense too get meaningful information. We will zoom to a year $1986$ to see the price dynamics. We also add a little addition. In the below plot, we plot money -- an abitrary one -- and we buy each commodities in equal. We will plot how our money grow or decline overtime.

# In[ ]:


plt.figure(figsize=(14,3))
plt.title("Price Change in Overtime")
plt.xlabel("Year")
plt.ylabel("Change of Price in Percent")
plt.grid()
plt.plot(df_train['bean_settle'].loc['1986'].pct_change())
plt.plot(df_train['Gold'].loc['1986'].pct_change())
plt.plot(df_train['USD'].loc['1986'].pct_change())
plt.legend(["Soybean", "Gold", "USD"])

# We find the change percentage, fill the NaN with 0, and sort the index
gold_price = df_train["Gold"].loc['1986'].pct_change().fillna(0).sort_index()
usd_price = df_train["USD"].loc['1986'].pct_change().fillna(0).sort_index()
soy_price = df_train["bean_settle"].loc['1986'].pct_change().fillna(0).sort_index()

final_array = []
for i in [gold_price, usd_price, soy_price]:
    init = 10000
    arr = []
    for k in i:
        init += init * k
        arr.append(init)
    final_array.append(arr)

plt.figure(figsize=(14,3))
plt.title(str(10000) + " Decline/Growth in Overtime")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid()
plt.plot(gold_price.index, final_array[0])
plt.plot(gold_price.index, final_array[1])
plt.plot(gold_price.index, final_array[2])
plt.legend(["Gold", "USD", "Soybean"])

gold_price = df_train["Gold"].loc[:'1986'].pct_change().fillna(0).sort_index()
usd_price = df_train["USD"].loc[:'1986'].pct_change().fillna(0).sort_index()
soy_price = df_train["bean_settle"].loc[:'1986'].pct_change().fillna(0).sort_index()
oil_price = df_train["Oil"].loc[:'1986'].pct_change().fillna(0).sort_index()

final_array = []
for i in [gold_price, usd_price, soy_price, oil_price]:
    init = 10000
    arr = []
    for k in i:
        init += init * k
        arr.append(init)
    final_array.append(arr)

plt.figure(figsize=(14,3))
plt.title(str(10000) + " Decline/Growth in Overtime")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid()
plt.plot(gold_price.index, final_array[0])
plt.plot(gold_price.index, final_array[1])
plt.plot(gold_price.index, final_array[2])
plt.plot(gold_price.index, final_array[3])
plt.legend(["Gold", "USD", "Soybean", "Oil"])


# Apparently, keeping soybean is better than keeping gold in 1986. Maybe not a real-physical soybean beacause it will rot overtime but a contract of ownership. In the long run, USD is the most stable among all commodities.  It is very interesting! 
# 
# Now, let's see about production and price. We sum all production data available and plot it with price. Since all production is binned in one year period, we need to bin the price in one year period. We can derive demand from production and prices of commodity. The first plot below is plot of price and production over time. If we multiply price and production we get demand. That is what second plot is. 

# In[ ]:


years = np.unique(df_train.index.year)
average_soybean_price = [df_train["bean_settle"].loc[str(i)].mean() for i in years]
total_soybean_production = [df_train[[production_origin[i+1] for i in range(0,22,2)]].sum(axis=1).loc[str(i)].mean() for i in years]

fig, ax1 = plt.subplots(figsize=(15,6))
ax1.set_title("PRICE AND SOYBEAN PRODUCTION")
ax1.plot(years, average_soybean_price, '-r')
ax1.set_xlabel('Year')
ax1.set_ylabel('Price')
ax1.tick_params('y')
ax1.legend(["PRICE"], loc=2)

ax2 = ax1.twinx()
ax2.plot(years, total_soybean_production)
ax2.set_ylabel('Soybean Production')
ax2.tick_params('y')
ax2.legend(["PRODUCTION"], loc=4)
ax1.grid()
ax2.grid(linestyle=":")

plt.figure(figsize=(15,6))
plt.title("Soybean Demand")
plt.xlabel("Year")
plt.ylabel("Demand")
plt.grid()
plt.plot(years, np.multiply(total_soybean_production, average_soybean_price))


# ### Soybean Pricing and its Derivative
# 
# Finally we want to map the OCHLV value of soybean and its derivative. There are two derivatives which is meal and oil. Each of these derivatives have their own OCHLV. We don't understand the relation between openint and the rest of OHLCV. Numbers in openint is too large for OHLCV plot.

# In[ ]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go

init_notebook_mode(connected=True) #do not miss this line

trace0 = go.Scatter(x = df_train.bean_settle.index,
                   y = df_train.bean_settle,
                   mode = 'lines',
                   name = 'Soybean')

trace1 = go.Scatter(x = df_train.bean_settle.index,
                   y = df_train.meal_settle,
                   mode = 'lines',
                   name = 'Soymeal')

trace2 = go.Scatter(x = df_train.bean_settle.index,
                   y = df_train.soyoil_settle,
                   mode = 'lines',
                   name = 'Soyoil')

data = [trace0, trace1, trace2]
fig = go.Figure(data=data)

py.offline.iplot(fig)


# That's all for now. Any feedback will be appreciated. 
