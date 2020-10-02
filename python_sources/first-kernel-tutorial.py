#!/usr/bin/env python
# coding: utf-8

# # Countries of the World
# 
# As it is my first kernel, there may be mistakes in it, for example the population in regions graph. So if you have an idea about changes, problem solving, shortcuts or suggestions please leave a comment.
# 
# **Content:**
# 1. [Diagnose Data](#1)
# 2. [Building & Manipulating DataFrames, Filtering, Missing Data, Visualization, Python Tricks](#2)
#     1. [Countries in the Regions](#3)
#     2. [Top 5 Most Populous Countries](#4)
#     3. [Population Density in  the Regions](#5)
#     4. [Population in the Regions](#6)
#     5. [Coastline](#7)
#     6. [Arrangement of  Features](#8)
#     7. [Heatmap](#9)
#     11. [Top 20 Countries by Gdp(per capita)](#13)
#     8. [Correlations between some features](#10)
#     9. [Crops in the Regions](#11)
#     10. [Countries with Increasing & Decreasing Population (Based on Birth Rate & Death Rate Features)](#12)
#     
#     
# 

# <a id="1"></a> <br>
# # Diagnose Data 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
init_notebook_mode(connected=True)    
import seaborn as sns
# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv("../input/countries of the world.csv")


# In[ ]:


data.info()


# In[ ]:


#data


# In[ ]:


data.describe() #only numeric features


# In[ ]:


data.head()


# In[ ]:


data.tail()


# <a id="2"></a> <br>
# # Building & Manipulating DataFrames, Filtering, Missing Data, Visualization, Python Tricks

# <a id="3"></a> <br>
# ### Countries in the Regions

# In[ ]:


a=data["Region"].value_counts(dropna=False)


# In[ ]:


print(a)


# In[ ]:


type(a)


# In[ ]:


df = a.rename(None).to_frame().T


# In[ ]:


df2 = a.rename(None).to_frame()


# In[ ]:


c=list(df.columns)


# In[ ]:


df


# In[ ]:


df2


# In[ ]:


b=df2[0].tolist()


# In[ ]:


print(type(b))
b


# In[ ]:


#dct={"Regions:":c,"Counts of Countries":[1,2,4]}
regions=c
coc=b


# In[ ]:


list_label=["Regions","Counts_of_Countries"]
list_col=[regions,coc]


# In[ ]:


zipped=list(zip(list_label,list_col))


# In[ ]:


data_dict=dict(zipped)


# In[ ]:


data_dict


# In[ ]:


dataf1=pd.DataFrame(data_dict)


# In[ ]:


dataf1


# In[ ]:


trace = go.Bar(
                x = dataf1.Regions,
                y = dataf1.Counts_of_Countries,
                name = "counts",
                marker = dict(color = 'rgb(100, 174, 255)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dataf1.Regions)
datag = [trace]
fig = go.Figure(data = datag)
py.offline.iplot(fig)


# <a id="4"></a> <br>
# ### Top 5 Most Populos Countries

# In[ ]:


filt=data.sort_values(by=['Population'],ascending=False)


# In[ ]:


#filt


# In[ ]:


filt.index=range(0,227,1)


# In[ ]:


dfgraph=filt.loc[0:5,["Country","Population"]]


# In[ ]:


dfgraph


# In[ ]:


trace1 = go.Bar(
                x = dfgraph.Country,
                y = dfgraph.Population,
                marker = dict(color = 'rgb(255, 0, 0)',
                             line=dict(color='rgb(0,0,0)',width=2.0)))
datag1 = [trace1]
fig = go.Figure(data = datag1)
py.offline.iplot(fig)


# <a id="5"></a> <br>
# ### Population Density in the Regions

# In[ ]:


area=data.groupby("Region")["Area (sq. mi.)"].sum()


# In[ ]:


#area


# In[ ]:


d=area.values.tolist()


# In[ ]:


len(d)
d


# In[ ]:


popl=data.groupby("Region")["Population"].sum()


# In[ ]:


popl


# In[ ]:


e=popl.values.tolist()
e


# In[ ]:


rp=[]
for i in range(0,11):
    num=(e[i]/d[i])
    rp.append("%.2f" %num) 


# In[ ]:


rp


# In[ ]:


regions=c
popdor=rp


# In[ ]:


list_label2=["Regions","Population_Density_of_Regions"]
list_col2=[regions,popdor]
zipped2=list(zip(list_label2,list_col2))
data_dict2=dict(zipped2)
dataf2=pd.DataFrame(data_dict2)


# In[ ]:


dataf2


# In[ ]:


#"%.2f" % 1.5768319


# In[ ]:


trace2 = go.Bar(
                x = dataf2.Regions,
                y = dataf2.Population_Density_of_Regions,
                name = "counts",
                marker = dict(color = 'rgb(110, 0, 0)',
                             line=dict(color='rgb(0,0,0)',width=2.0)),
                text = dataf1.Regions)
datag2 = [trace2]
fig = go.Figure(data = datag2)
py.offline.iplot(fig)


# <a id="6"></a> <br>
# ### Population in the Regions

# In[ ]:


list_label1=["Region","Population"]
list_col1=[regions,e]
zipped1=list(zip(list_label1,list_col1))
data_dict1=dict(zipped1)
dataf3=pd.DataFrame(data_dict1)


# In[ ]:


dataf3


# In[ ]:


#some issues in this graph(in SUB-SAHARAN AFRICA population value)
trace3 = go.Bar(
                x = dataf3.Region,
                y = dataf3.Population,
                name = "counts",
                marker = dict(color = 'rgb(110, 95, 0)',
                             line=dict(color='rgb(0,0,0)',width=2.0)))
datag3 = [trace3]
fig3 = go.Figure(data = datag3)
py.offline.iplot(fig3)


# <a id="7"></a> <br>
# ### Coastline

# In[ ]:


comp=data["Coastline (coast/area ratio)"].tolist()


# In[ ]:


arrang=[]
for i in data["Coastline (coast/area ratio)"]:
    arrang.append(i.split(",")[0]+"."+i.split(",")[1])


# In[ ]:


data["Coastline (coast/area ratio)"]=arrang
data["Coastline (coast/area ratio)"]=data["Coastline (coast/area ratio)"].astype("float")
data.info()


# In[ ]:


filt1=data["Coastline (coast/area ratio)"]>0


# In[ ]:


sumcoast=filt1.value_counts().tolist()


# In[ ]:


trace4 = go.Bar(
                x = ["Number of Countries with Coastline","Number of Countries without Coastline"],
                y = sumcoast,
                marker = dict(color = 'rgb(110, 95, 147)',
                             line=dict(color='rgb(0,0,0)',width=2.0)))
datag4 = [trace4]
fig4 = go.Figure(data = datag4)
py.offline.iplot(fig4)


# <a id="8"></a> <br>
# ### Arrangement of  Features

# In[ ]:


data.info() #Below you can see which features have NaN values.


# In[ ]:


#data["Infant mortality (per 1000 births)"]


# **Some features need to be edited to use astype() method. So we must turn commas to points.**

# In[ ]:


arrang1=[]
example="example"
for i in data["Infant mortality (per 1000 births)"]:
    if((type(i))==type(example)):
        if(len(i.split(","))>1):
            type(i)
            arrang1.append(i.split(",")[0]+"."+i.split(",")[1])
        else:
            arrang1.append(i)
    else: #NaN is float variable.
        arrang1.append(i)


# In[ ]:


len(arrang1)


# In[ ]:


data["Infant mortality (per 1000 births)"]=arrang1


# In[ ]:


data["Infant mortality (per 1000 births)"]=data["Infant mortality (per 1000 births)"].astype("float")


# In[ ]:


#data["Infant mortality (per 1000 births)"]


# In[ ]:


data["Infant mortality (per 1000 births)"].mean()


# In[ ]:


data["Infant mortality (per 1000 births)"].fillna(data["Infant mortality (per 1000 births)"].mean(),inplace=True)


# In[ ]:


#data["Infant mortality (per 1000 births)"]


# In[ ]:


#data["Literacy (%)"]


# In[ ]:


arrang2=[]
for i in data["Literacy (%)"]:
    if((type(i))==type(example)):
        if(len(i.split(","))>1):
            type(i)
            arrang2.append(i.split(",")[0]+"."+i.split(",")[1])
        else:
            arrang2.append(i)
    else:
        arrang2.append(i)


# In[ ]:


arrang2
len(arrang2)


# In[ ]:


data["Literacy (%)"]=arrang2
data["Literacy (%)"]=data["Literacy (%)"].astype("float")
data["Literacy (%)"].fillna(data["Literacy (%)"].mean(),inplace=True)


# In[ ]:


data.info()
data["GDP ($ per capita)"].fillna(data["GDP ($ per capita)"].mean(),inplace=True )


# In[ ]:


#data["Industry"]


# In[ ]:


arrang3=[]
for i in data["Industry"]:
    if((type(i))==type(example)):
        if(len(i.split(","))>1):
            type(i)
            arrang3.append(i.split(",")[0]+"."+i.split(",")[1])
        else:
            arrang3.append(i)
    else:
        arrang3.append(i)


# In[ ]:


arrang3
len(arrang3)


# In[ ]:


data["Industry"]=arrang3
data["Industry"]=data["Industry"].astype("float")
data["Industry"].fillna(data["Industry"].mean(),inplace=True)


# In[ ]:


data.info()


# In[ ]:


data["Agriculture"]
arrang4=[]
for i in data["Agriculture"]:
    if((type(i))==type(example)):
        if(len(i.split(","))>1):
            type(i)
            arrang4.append(i.split(",")[0]+"."+i.split(",")[1])
        else:
            arrang4.append(i)
    else:
        arrang4.append(i)


# In[ ]:


arrang4
len(arrang4)
data["Agriculture"]=arrang4
data["Agriculture"]=data["Agriculture"].astype("float")
data["Agriculture"].fillna(data["Agriculture"].mean(),inplace=True)


# In[ ]:


data["Deathrate"]
arrang5=[]
for i in data["Deathrate"]:
    if((type(i))==type(example)):
        if(len(i.split(","))>1):
            type(i)
            arrang5.append(i.split(",")[0]+"."+i.split(",")[1])
        else:
            arrang5.append(i)
    else:
        arrang5.append(i)
data["Birthrate"]
arrang6=[]
for i in data["Birthrate"]:
    if((type(i))==type(example)):
        if(len(i.split(","))>1):
            type(i)
            arrang6.append(i.split(",")[0]+"."+i.split(",")[1])
        else:
            arrang6.append(i)
    else:
        arrang6.append(i)


# In[ ]:


#len(arrang5)
#len(arrang6)


# In[ ]:


data["Deathrate"]=arrang5
data["Deathrate"]=data["Deathrate"].astype("float")
data["Deathrate"].fillna(data["Deathrate"].mean(),inplace=True)
data["Birthrate"]=arrang6
data["Birthrate"]=data["Birthrate"].astype("float")
data["Birthrate"].fillna(data["Birthrate"].mean(),inplace=True)


# In[ ]:


data["Crops (%)"]
arrang7=[]
for i in data["Crops (%)"]:
    if((type(i))==type(example)):
        if(len(i.split(","))>1):
            type(i)
            arrang7.append(i.split(",")[0]+"."+i.split(",")[1])
        else:
            arrang7.append(i)
    else:
        arrang7.append(i)


# In[ ]:


data["Crops (%)"]=arrang7
data["Crops (%)"]=data["Crops (%)"].astype("float")
data["Crops (%)"].fillna(data["Crops (%)"].mean(),inplace=True)


# <a id="13"></a> <br>
# ### Top 20 Countries by Gdp (per capita)

# In[ ]:


filt3=data.sort_values(by=['GDP ($ per capita)'],ascending=False)


# In[ ]:


#filt3


# In[ ]:


filt3.index=range(0,227,1)
filt3graph=filt3.loc[0:20,["Country","GDP ($ per capita)"]]
filt3graph


# In[ ]:


trace11 = go.Bar(
                x = filt3graph.Country,
                y = filt3graph["GDP ($ per capita)"],
                marker = dict(color = 'rgb(229, 165, 224)',
                             line=dict(color='rgb(0,0,0)',width=2.0)))
datag11 = [trace11]
fig11 = go.Figure(data = datag11)
py.offline.iplot(fig11)


# <a id="9"></a> <br>
# ### Heatmap

# In[ ]:


data.corr()


# In[ ]:


f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
# I think this correlation a little bit confusing because of GDP-Industry relations.


# <a id="10"></a> <br>
# ### Correlations between some features

# In[ ]:


trace7 = go.Scatter(
    x = data["GDP ($ per capita)"],
    y = data["Agriculture"],
    mode = 'markers'
)

data7 = [trace7]

# Plot and embed in ipython notebook!
py.offline.iplot(data7)


# In[ ]:


#confusing because feature has a percentage value
trace8 = go.Scatter(
    x = data["GDP ($ per capita)"],
    y = data["Literacy (%)"],
    mode = 'markers'
)

data8 = [trace8]

# Plot and embed in ipython notebook!
py.offline.iplot(data8)


# In[ ]:


trace9 = go.Scatter(
    x = data["GDP ($ per capita)"],
    y = data["Industry"],
    mode = 'markers'
)

data9 = [trace9]

# Plot and embed in ipython notebook!
py.offline.iplot(data9)


# In[ ]:


trace10 = go.Scatter(
    x = data["GDP ($ per capita)"],
    y = data["Infant mortality (per 1000 births)"],
    mode = 'markers'
)

data10 = [trace10]

# Plot and embed in ipython notebook!
py.offline.iplot(data10)


# <a id="11"></a> <br>
# ### Crops in the Regions

# In[ ]:


data["Crops (%)"]


# In[ ]:


crop=data["Area (sq. mi.)"].values.tolist()
len(crop)


# In[ ]:


cropper=data["Crops (%)"].values.tolist()
len(cropper)
#cropper


# In[ ]:


cropf=[]
for i in range(0,227):
    cropf.append("%.2f" % (crop[i]*cropper[i]))


# In[ ]:


len(cropf)
data["Crops_Field (sq.mi.)"]=cropf
data["Crops_Field (sq.mi.)"]=data["Crops_Field (sq.mi.)"].astype("float")


# In[ ]:


data.head()


# In[ ]:


cropsum=data.groupby("Region")["Crops_Field (sq.mi.)"].sum()


# In[ ]:


cropsuml=cropsum.tolist()
cropsuml


# In[ ]:


trace5 = go.Bar(
                x = c,
                y = cropsuml,
                marker = dict(color = 'rgb(229, 165, 121)',
                             line=dict(color='rgb(0,0,0)',width=2.0)))
datag5 = [trace5]
fig5 = go.Figure(data = datag5)
py.offline.iplot(fig5)


# <a id="12"></a> <br>
# ### Countries with Increasing & Decreasing Population (Based on Birth Rate & Death Rate Features)

# In[ ]:


br=data["Birthrate"].values.tolist()
dr=data["Deathrate"].values.tolist()


# In[ ]:


sumbrdr=[]
for i in range(0,227):
    sumbrdr.append("%.2f" %(br[i]-dr[i]))


# In[ ]:


#sumbrdr


# In[ ]:


data["Birthrate-Deathrate"]=sumbrdr
data["Birthrate-Deathrate"]=data["Birthrate-Deathrate"].astype("float")


# In[ ]:


data.head()


# In[ ]:


filt2=data["Birthrate-Deathrate"]>0


# In[ ]:


brdr=filt2.value_counts().tolist()


# In[ ]:


trace6 = go.Bar(
                x = ["Countries with Increasing Population","Countries with Decreasing Population"],
                y = brdr,
                marker = dict(color = 'rgb(229, 228, 121)',
                             line=dict(color='rgb(0,0,0)',width=2.0)))
datag6 = [trace6]
fig6 = go.Figure(data = datag6)
py.offline.iplot(fig6)


# In[ ]:




