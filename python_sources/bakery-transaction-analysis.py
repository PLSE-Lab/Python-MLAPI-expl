#!/usr/bin/env python
# coding: utf-8

# ### Transactions from a bakery - Market Basket Analysis - Kaggle
# **Data set** <br>
# - Date: 2016-10-30  to 2017-04-09 <br>
# - Time<br>
# - Transaction<br>
# - Item <br>
# 
# **Questions to explore** <br>
# Check out my app visualizing this data set at https://utn100-marketbasket.herokuapp.com/ 
# <br>
# [I.Data exploratory](#I)
#     - Busiest times of the day are from 9am-3pm, peaks around 10 am
#     - Amount of transactions increases toward the end of week, peaks on Saturday
#     - Among months that have full data (Nov-Mar), November has the highest number of transaction, but in general, there is not significant variation through months.
#     - Examples of most popular items sold are coffee, bread, pastry, and cake
#     
# [II. Market Basket Analysis](#II) <br>
# Analyzing association of items: market basket analysis. Codes were modified from http://intelligentonlinetools.com/blog/2018/02/10/how-to-create-data-visualization-for-association-rules-in-data-mining/
# 
# [III.Time series prediction of daily number of  transactions](#III)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import seaborn as sns
import random
from datetime import datetime
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# ## I-Exploratory Data Analysis <a class="anchor" id="I"></a>

# In[ ]:


# Import data
df = pd.read_csv("../input/BreadBasket_DMS.csv")
print(df.head())
print(df.info())
# Check unique item categories
print("No of unique item: {}".format(df['Item'].nunique()))
print("\n")
print("Top items and counts:","\n", df['Item'].value_counts().head(5))


# In[ ]:


# Drop row of NONE items
df.drop(df[df['Item']=='NONE'].index,inplace=True)

# Create month and hour columns
df['Month'] = df['Date'].apply(lambda x:x.split("-")[1])
df['Hour'] = df['Time'].apply(lambda x:int(str(pd.to_datetime(x).round('H')).split(" ")[1].split(":")[0]))

#Busiest hours
df.groupby('Hour')['Transaction'].nunique().plot(figsize=(8,5))
plt.title("Hourly number of transactions")


# In[ ]:


#Busiest day of week
#Get total transaction for each date
byday=df.groupby('Date')['Transaction'].nunique().reset_index()
#Create dayofweek column
byday['DayofWeek'] = byday['Date'].apply(lambda x:pd.to_datetime(x).weekday())

#Plot average transactions per day
byday.groupby('DayofWeek')['Transaction'].mean().plot(figsize=(8,5))
plt.title("Average number of transactions per weekday")
plt.ylabel("# of Transactions")
fig=plt.xticks(np.arange(7),['Mon','Tue','Wed','Thur','Fri','Sat','Sun'])


# In[ ]:


# Busiest months
bymonth=df.groupby("Month")['Transaction'].nunique().reset_index()
bymonth['Month'] = ['Jan','Feb','Mar','Apr','Oct','Nov','Dec']
bymonth['Order'] = [4,5,6,7,1,2,3]
bymonth.sort_values(by='Order',inplace=True)
plt.plot(bymonth['Month'],bymonth['Transaction'])
plt.title("Monthly number of transactions")


# In[ ]:


# Daily transaction
df.groupby("Date")['Transaction'].nunique().plot(figsize=(12,6),label="Daily number of transaction")
df.groupby("Date")['Transaction'].size().plot(figsize=(12,6),label="Daily number of items sold")
plt.legend(loc='best')


# In[ ]:


# Get counts of each item per hour for each months
byitem=df.groupby(["Month","Hour",'Item']).size().reset_index().sort_values(by="Hour")
byitem.rename(columns={0:'Total'},inplace=True)
byitem.head()


# In[ ]:


item_toplot = ['Coffee','Bread','Tea','Pastry','Cake']
for item in item_toplot:
    byitem[byitem['Item']==item].groupby("Hour")['Total'].mean().plot(label=item,figsize=(8,5))

plt.legend(loc='best')
plt.title("Average items sold per hour")


# In[ ]:


df_agg = byitem.groupby(['Month','Hour','Item']).agg({'Total':sum})
g = df_agg['Total'].groupby(level=[0,1], group_keys=False)
topitem_hour=g.nlargest(5).reset_index()
topitem_hour.head(10)


# ## II-Market Basket Analysis <a class="anchor" id="II"></a>

# In[ ]:


# Get the item basket lists for all transactions
items = []
for i in df['Transaction'].unique():
    itemlist = list(set(df[df["Transaction"]==i]['Item']))
    if len(itemlist) > 0:
        items.append(itemlist)
    
print(items[0:5])
print(len(items))


# In[ ]:


tran_encoder = TransactionEncoder()
oht_ary = tran_encoder.fit(items).transform(items)
dataframe = pd.DataFrame(oht_ary, columns=tran_encoder.columns_)
print (dataframe.head())           
 
frequent_itemsets = apriori(dataframe, use_colnames=True, min_support=0.02)
print (frequent_itemsets.head())
 
#association_rules(frequent_itemsets, metric="confidence", min_threshold=0.05)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
print (rules.head())


# In[ ]:


# Scatter plot of support vs. confidence
plt.figure(figsize=(8,5))
plt.scatter(rules['support'],rules['confidence'],marker='*',edgecolors='grey',s=100,c=rules['lift'],cmap='rainbow')
plt.colorbar(label='Lift')
plt.xlabel('support')
plt.ylabel('confidence') 


# In[ ]:


import networkx as nx  
rules_to_show = len(rules)
G1 = nx.DiGraph()
    
color_map=[]
N = 400
colors = np.random.rand(N)    
strs=[]
for i in range(rules_to_show):
    strs.append('R'+str(i))
    
for i in range (rules_to_show):      
    G1.add_nodes_from(["R"+str(i)])         
        
    for a in rules.iloc[i]['antecedents']:                
        G1.add_nodes_from([a])        
        G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 1.5)
        
    for c in rules.iloc[i]['consequents']:         
        G1.add_nodes_from([c])
        G1.add_edge("R"+str(i), c, color=colors[i],  weight=1.5)
    
for node in G1:
    if node in strs:
        color_map.append('yellow')
    else:
        color_map.append('pink')
            
edges = G1.edges()
colors = [G1[u][v]['color'] for u,v in edges]
weights = [G1[u][v]['weight'] for u,v in edges]


# In[ ]:


# modified positions of nodes
# Get coordinates in circle
def PointsInCircum(r,n=20):
    import math
    return [math.cos(2*math.pi/n*x)*r for x in np.arange(0,n+1)], [math.sin(2*math.pi/n*x)*r for x in np.arange(0,n+1)]
cir_x,cir_y = PointsInCircum(40,rules_to_show*2)[0],PointsInCircum(40,rules_to_show*2)[1]

pos = {}
index1 = int(rules_to_show/4)+1
index2 = -(int(rules_to_show/4)+1)
for node in G1:
    if node not in strs:
        pos[node]= [cir_x[index1],cir_y[index1]]
        index1 += 1 
    else:
        pos[node]= [cir_x[index2],cir_y[index2]]
        index2 -=1
plt.figure(figsize=(12,10))
nx.draw(G1, pos, edges=edges, node_color = color_map, edge_color=colors, width=weights, font_size=20, with_labels=False)            

for p in pos:  # raise text positions
    pos[p][1] += 3

fig=nx.draw_networkx_labels(G1, pos)


# ## III-Time Series prediction of daily number of transactions <a class="anchor" id="III"></a>

# In[ ]:


import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA

transac = df.groupby("Date")['Transaction'].nunique().reset_index()
transac.set_index("Date",inplace=True)
transac.index = pd.to_datetime(transac.index)
transac['Transaction']=transac['Transaction'].astype(float)

transac['7-day-SMA'] = transac['Transaction'].rolling(window=7).mean()
transac.plot(figsize=(12,6))


# In[ ]:


decomposition = seasonal_decompose(transac['Transaction'], freq=7)  
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)


# In[ ]:


# Store in a function for later use!
def adf_check(time_series):
    """
    Pass in a time series, returns ADF report
    """
    result = adfuller(time_series)
    print('Augmented Dickey-Fuller Test:')
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']

    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


# In[ ]:


adf_check(transac['Transaction'])


# In[ ]:


transac['First Difference'] = transac['Transaction'] - transac['Transaction'].shift(1)
transac['First Difference'].plot()


# In[ ]:


transac['Seasonal Difference'] = transac['First Difference'] - transac['First Difference'].shift(7)
transac['Seasonal Difference'].plot()


# In[ ]:


adf_check(transac['Seasonal Difference'].dropna())


# In[ ]:


from pandas.plotting import autocorrelation_plot
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(transac['Seasonal Difference'].iloc[13:], lags=20, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(transac['Seasonal Difference'].iloc[13:], lags=20, ax=ax2)


# In[ ]:


model = sm.tsa.statespace.SARIMAX(transac['Transaction'],order=(1,0,0), seasonal_order=(1,1,1,7))
results = model.fit()
print(results.summary())


# In[ ]:


transac['forecast'] = results.predict(start = 120, end= 158, dynamic= True)  
transac[['Transaction','forecast']].plot(figsize=(12,8))


# In[ ]:


from pandas.tseries.offsets import DateOffset
future_dates = [transac.index[-1] + DateOffset(n=x) for x in range(0,100) ]
future_dates_df = pd.DataFrame(index=future_dates[1:],columns=transac.columns)
future_df = pd.concat([transac,future_dates_df])


# In[ ]:


future_df['forecast'] = list(future_df['forecast'][0:159])+list(results.predict(start=158,end=256))
future_df[['Transaction', 'forecast']].plot(figsize=(12, 8)) 


# In[ ]:




