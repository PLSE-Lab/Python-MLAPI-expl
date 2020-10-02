#!/usr/bin/env python
# coding: utf-8

# ## Loading libraries and data

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot,plot
init_notebook_mode(connected=True)
import  plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import numpy as np

df1=pd.read_csv("../input/world-foodfeed-production/FAO.csv",encoding="ISO-8859-1")
df1=df1.copy()
df3=pd.read_csv("../input/gdp-world-bank-data/GDP by Country.csv",skiprows=3)
df3=df3.copy()


# # What are the top products produced around the globe

# In[ ]:



plt.style.use("ggplot")
items=pd.DataFrame(df1.groupby("Item")["Element"].agg("count").sort_values(ascending=False))[:25]
plt.figure(figsize=(8,10))
plt.gcf().subplots_adjust(left=.3)
sns.barplot(x=items.Element,y=items.index,data=items)
plt.gca().set_title("Top 25 items produced around globe")
plt.show()


# ## India's production story

# In[ ]:



india=pd.DataFrame(df1[df1["Area"]=="India"].loc[:,"Y2004":"Y2013"].agg("sum",axis=0))
plt.figure(figsize=(8,8))

india.columns=["Production"]
sns.barplot(x=india.index,y=india.Production,data=india)
plt.gca().set_ylabel("Production in tonnes")
plt.gca().set_xlabel("Year")
plt.gcf().subplots_adjust(left=.2)
plt.gca().set_title("India's production")
plt.show()


# ## Which are the top producers throughout the years

# In[ ]:


top_countries=df1.groupby(["Area"])[["Y2004","Y2005","Y2005","Y2006","Y2007","Y2008","Y2009","Y2010","Y2011",
                                    "Y2012","Y2013"]].sum()
top=pd.DataFrame(top_countries.agg("mean",axis=1).sort_values(ascending=False),columns=["Tonnes"])[:10]


# In[ ]:


plt.figure(figsize=(8,8))
plt.gca().set_title("Top producers throughout the years")
sns.barplot(x=top["Tonnes"],y=top.index,data=top)
plt.gcf().subplots_adjust(left=.3)
plt.show()


# In[ ]:





# 
# ## Variation in production of top 4 producers

# In[ ]:



india=pd.DataFrame(top_countries.loc["India"])
china=pd.DataFrame(top_countries.loc["China, mainland"])
us=pd.DataFrame(top_countries.loc["United States of America"])
brazil=pd.DataFrame(top_countries.loc["Brazil"])

trace_1=go.Scatter(x=us.index,y=us["United States of America"],mode="lines",name="US")
trace_2=go.Scatter(x=india.index,y=india["India"],mode="lines+markers",name="India")
trace_3=go.Scatter(x=china.index,y=china["China, mainland"],mode="lines",name="China")
trace_4=go.Scatter(x=brazil.index,y=brazil["Brazil"],mode="lines+markers",name="Brazil")

data1=[trace_1,trace_2,trace_3,trace_4]

layout = dict(title = 'Variations in production of top 4 countries',
              xaxis= dict(title= 'Year',ticklen= 5,zeroline= False)
             )
fig=dict(data=data1,layout=layout)
iplot(fig,filename="figure.html",validate=False)


# ## Wheat production and price rise in India

# In[ ]:


df2=pd.read_csv("../input/global-food-prices/wfp_market_food_prices.csv",encoding="ISO-8859-1")
df2=df2.copy()


# In[ ]:


india=df2[(df2["adm0_name"]=="India")& (df2["mp_year"]>2003)]
india=pd.DataFrame(india.groupby(["cm_name","mp_year"])["mp_price"].agg("mean").reset_index())
wheat=df1[(df1["Area"]=="India") & (df1["Item"]=="Wheat and products")& (df1['Element']=="Food")]
india=india[(india["cm_name"]=="Wheat") & (india["mp_year"]<2014)]
wheat=wheat.loc[:,"Y2004":].transpose()
wheat.columns=["tonnes"]


# In[ ]:



plt.style.use("ggplot")
fig,ax1=plt.subplots(figsize=(8,6))
ax2=ax1.twinx()
x=np.arange(0,10)
y=india["mp_price"]
ax2.plot(x,y,'-o',color="red",label="price")

y=wheat["tonnes"]
ax1.plot(x,y,"-o",color="blue",label="production")
ax1.set_xticklabels(wheat.index,rotation="90")
ax2.set_xticklabels(wheat.index,rotation="90")
ax1.set_ylabel("production in tonnes")
ax2.set_ylabel("price in RS")
ax1.set_title("Production and price of Wheat in India")
plt.legend()
plt.show()


# 
# ## Food and Feed

# In[ ]:


foo=df1.groupby(["Element"]).agg("sum").drop(["Area Code","Item Code","Element Code","latitude","longitude"],axis=1)
foo=foo.sum(axis=1)
labels=["Feed","Food"]
values=[foo[0],foo[1]]

trace=go.Pie(labels=labels,values=values)
iplot([trace],filename="food_feed",validate=False)


# 
# ## Boxplot of top producers around globe

# In[ ]:


top_box=df1.groupby("Area").agg("sum").loc[top.index].drop(["Area Code","Item Code",
                                                            "latitude","longitude","Element Code"],axis=1)
top_box=top_box.transpose()
plt.style.use("ggplot")
plt.figure(figsize=(15,8))
sns.boxplot(data=top_box)
plt.show()


# ## Heatmap of top 5 Producers

# In[ ]:


plt.figure(figsize=(12,9))
top=pd.DataFrame(top_countries.agg("mean",axis=1).sort_values(ascending=False),columns=["Tonnes"])[:5]
top_box=df1.groupby("Area").agg("sum").loc[top.index].drop(["Area Code","Item Code",
                                                            "latitude","longitude","Element Code"],axis=1)
top_box=top_box.transpose()
sns.heatmap(data=top_box,linewidths=0,cmap="YlGnBu")
plt.show()


# In[ ]:


#It is evident from the heatmap that the China and India is constatly improving their production,but Brazil and USA 
#seems to have constant production throughout the year or has not much improved production compared to China or India.
#Russian Frederation was established on year 1991,so production till 1991 remains zero


# 
# 
# 
# ## India's production and gdp growth

# In[ ]:


india_gdp=df3[df3["Country Name"]=="India"].loc[:,"1961":"2004"].transpose()
india_gdp.columns=["gdp"]
india_pr=df1[df1["Area"]=="India"].loc[:,"Y1961":"Y2004"].agg("sum")

fig,ax1=plt.subplots(figsize=(15,7))
ax2=ax1.twinx()
x=india_gdp.index
ax2.plot(x,india_gdp["gdp"],"-",color="red",label="gdp")
ax1.plot(x,india_pr[:],"-",color="blue",label="production")
ax1.set_xticklabels(india_gdp.index,rotation="90")
ax1.set_ylabel("production in tonnes")
ax2.set_ylabel("GDP in USD")
fig.legend()
ax1.set_title("India's production and gdp growth",fontsize=25)

plt.show()


# In[ ]:


#Agriculture sector is one of the majaor contributer to india's gdp.


# ## Least Food producing countries

# In[ ]:


country=df1[df1["Element"]=="Food"].groupby("Area").agg("sum").drop(["Area Code","Item Code","Element Code","latitude","longitude"],axis=1)
country=pd.DataFrame(country.mean(axis=1).sort_values(),columns=["production"])[:20]
plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
sns.barplot(y=country.index,x=country["production"],data=country,palette="Set3")
plt.gca().set_xlabel("Mean production in tonnes",fontsize=15)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().set_title("Least Food producing countries",fontsize=25)
plt.show()
plt.savefig("least_food")


# In[ ]:




