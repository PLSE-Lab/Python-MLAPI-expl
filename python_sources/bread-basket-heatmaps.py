#!/usr/bin/env python
# coding: utf-8

# **To begin:**
# *
# This is my second time ever working with pandas so there may be a decent amount of unnecessary code. A lot of these lines were quick practices that I forgot to clean up.*
# 
# Anyways, I am interested in seeing how the flow of purchases occurs over time at this bakery. The first thing was to standardize the time points by half hour for ease of use later on. 

# In[ ]:


import pandas as pd
df =pd.read_csv('../input/BreadBasket_DMS.csv')
df2 = df['Time'].str[0:5].str.split(':',n=1,expand=True)
df2[[0,1]] = df2[[0,1]].apply(pd.to_numeric)
df2[1] = df2[1]*10/6
highmask = df2[1] > 50
lowmask = df2[1] < 50
df2.loc[highmask, 1] = 50
df2.loc[lowmask, 1] = 0
df2[0] = (df2[0].astype(str)+'.'+df2[1].astype(str).str[0:1]).astype(float)
df = pd.concat([df,df2], axis=1, sort=False)
df.pop(1)
df.head()


# In[ ]:


timelist = df[0].unique().tolist()
#df[0].value_counts()


# Next I made a list of all of the item names and proceeded to get some standard statistical info about the transactions of each item (average time purchased, number of transactions, etc.) and made a new dataframe from this.

# In[ ]:


allitems = df['Item'].unique().tolist()


# In[ ]:


newdic = {}
for i in range(len(allitems)):
    filt = df['Item'].str.contains(allitems[i])
    newdf = df[filt]
    meandf = newdf['Time'].str[0:5].str.split(':',n=1,expand=True)
    meandf[[0,1]] = meandf[[0,1]].apply(pd.to_numeric)
    meandf[1] = meandf[1]*10/6
    meandf[0] = (meandf[0].astype(str)+'.'+meandf[1].astype(str).str[0:1]).astype(float)
    count = meandf[0].count()
    mean = meandf[0].mean()
    std = meandf[0].std()
    newdic[allitems[i]]=(mean,std,count)


# In[ ]:


Results = pd.DataFrame(newdic, index=('Mean Time', 'Time STD', 'Transaction Count')).T


# In[ ]:


print(Results.sort_values(by=['Transaction Count'], ascending=False).head())


# I then used the data from that to determine what the top 25 items purchased were and then put them in order based on when they were bought most.

# In[ ]:


orderresults = Results.sort_values(by=['Mean Time'], ascending=True)
orderlist = orderresults.index.values
top = Results.sort_values(by='Transaction Count', ascending=False)
toplist= top.index.values
toplist = toplist[:26]
toplist
orderedtop = []
p = 0
while p < len(orderlist):
    if orderlist[p] in toplist:
        orderedtop.append(orderlist[p])
    p+=1
orderedtop.remove('NONE')


# For the purposes of a later graph I made up prices for each item. Maybe there is a menu online somewhere I could get real numbers, but I gave it my best guess.

# In[ ]:


orderedtopprice = {'Pastry':2,'Toast':1.50,'Medialuna':2,'Baguette':3,'Farm House':4,'Bread':1.50,'Coffee':1,'Scandinavian':2,'Jam':0.50,'Muffin':2,'Spanish Brunch':12,'Cookies':2,'Juice':2,'Scone':2,'Hot chocolate':3,'Fudge':5,'Tea':1.50,'Brownie':2,'Tiffin':1,'Sandwich':4.5,'Alfajores':3,'Cake':3,'Soup':3,'Truffles':3,'Coke':1}


# I then made a new dataframe containing the number of purchases for each item at every time window found (eg. 10:00-10:30)

# In[ ]:


finaldict = {}
df4 = pd.DataFrame()
dfmean = pd.DataFrame()
i = 0
while i < len(allitems):
    filt = df['Item'].str.contains(allitems[i])
    df3 = df[filt]
    newseries = pd.Series(df3[0].value_counts(),name=allitems[i])
    average = newseries.max()
    meanseries = newseries/average
    df4 = pd.concat([df4,newseries], axis=1, sort=False)
    dfmean = pd.concat([dfmean,meanseries],axis=1,sort=False)
    i+=1

    


# Got rid of extraneous values

# In[ ]:


df5 = df4.drop([1.0,7.0,7.5,19.0,18.5,19.5,20.0,21.5,22.0,22.5,20.5,23.0,23.5])
dfmean = dfmean.drop([1.0,7.0,7.5,19.0,18.5,19.5,20.0,21.5,22.0,22.5,20.5,23.0,23.5])


# Reordered the dataframe based on average time bought

# In[ ]:


df6 = df5[orderedtop]
dfmean = dfmean[orderedtop]


# And then made some heatmaps! Time is the bottom axis (in military). The first map shows when each of the top 25 items was bought most frequently, with one being the highest amount and 0 being the lowest. This helps show the trend of sales throughout the day. You can see that toast is bought in the morning, sandwiches are bought in the afternoon, coffee is bought all day, and cookies are bought during only breakfast and lunch time.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
dfmean = dfmean.fillna(0)
heatmap = dfmean.T
df6 = df6.fillna(0)
heatmap2 = df6.T
plt.subplots(figsize=(20,15))
sns.heatmap(heatmap, annot=False, fmt="g", cmap='viridis')


# This map then shows the overall transaction number, all it really says is that coffee and bread are by far the most sold items.

# In[ ]:


plt.subplots(figsize=(20,15))
sns.heatmap(heatmap2, annot=False, fmt="g", cmap='viridis')


# But that doesn't really matter if coffee only earns you 50 cents while a sandwich earns you 5 bucks! So here is the heatmap of revenue by item throughout the day based on what I assumed the prices for each item was.  From this you can see what items to focus on at what times of day and how you could adjust prices in a worthwhile way. In addition, you could think of certain deals like a coffee and a cookie for 10% off to take advantage of the trends seen here. 

# In[ ]:


df7 =df6
for i in range(len(orderedtop)):
    df7[orderedtop[i]] = df7[orderedtop[i]]*orderedtopprice[orderedtop[i]]
priceadjusted = df7.T
plt.subplots(figsize=(20,15))
sns.heatmap(priceadjusted, annot=False, fmt="g", cmap='viridis')


# In[ ]:





# In[ ]:





# In[ ]:




