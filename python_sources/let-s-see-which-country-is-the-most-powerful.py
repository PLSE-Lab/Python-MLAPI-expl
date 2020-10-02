#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
from io import StringIO
import cufflinks
import plotly.graph_objs as go


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
init_notebook_mode(connected=True)
sns.set_style('whitegrid')
cufflinks.go_offline()
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


me = pd.read_csv('../input/military-expenditure-of-countries-19602019/Military Expenditure.csv')


# Let have a quick check the data

# In[ ]:


me.head()


# In[ ]:


me.info()


# There are a lot of NA in this dataset, I will check whether there is any negative number. If not, I will fillna with 0

# In[ ]:


a = me[me.columns[4:]]<0
s = a.any(1)
s[s==True]


# In[ ]:


me.fillna(value=0,inplace=True)


# <p>Ok, so now I will only want to see the Data with Country Name and Type = Country. </p>
# <p> I will set the index to Country Name to visualize it on graph </p>

# In[ ]:


data = me[me['Type'].apply(lambda x: x in ['Country'])] .drop(['Code','Type','Indicator Name'],axis=1).set_index('Name')


# In[ ]:


plt.figure(figsize=(14,8))
sns.heatmap(data)


# <p> We can see that the US has invested a large amount of money into Military, followed by Brazil </p>
# <p> Let figure out Top 10 country in Military Expenditure </p>

# In[ ]:


top10 = data.sum(axis=1).sort_values(ascending=False)[:10]
data_index = top10.index
data_top10 = data.loc[data_index.tolist()]


# In[ ]:


plt.figure(figsize=(14,8))
sns.heatmap(data_top10)


# <p> OK I think heatmap is not good in this situation, let try in linegraph</p>
# <p> So top 10 in the world: USA, China, France, UK, Germany, Japan, Saudi Arabia, Russian Federation, India and Italy</p>
# <p> Where is Brazil :-?</p>

# In[ ]:


data_top10.transpose().iplot(kind='line')


# <p>If you remove USA from the list, China has just put a lot of the money to military since 2005 and in 2007 it became the 2nd country that have the most military expenditure</p>
# <p>Regarding the rest, I noticed Russia and Saudi Arabia, They started at the bottom level and now in the top 10 of the World </p>
# <p> Hold on Russian started to invest on military in 1991? May be in the past its name was Soviet,let me check later</p>

# Now I am living in Canada so I will check the amount of money for soldier in North America. But the US is an outliner so I will pick another country in the South of America, Brazil

# In[ ]:


data.reindex(['Canada','Brazil', 'Mexico']).transpose().iplot(kind='line')


# Yes, if we count only 2018, Brazil will be in Top 10 of the World. Before this, I just expected Canada didn't invest much money for military (but I was wrong, 21B is a quite large number)

# Ok, my home country is Vietnam so let check the situation in South East Asia

# In[ ]:


sea=['Brunei Darussalam', 'Cambodia', 'Indonesia', 'Laos', 'Malaysia', 'Myanmar', 'Philippines', 
     'Singapore', 'Thailand', 'Timor-Leste', 'Vietnam']
data_sea = data.reindex(sea)


# In[ ]:


top_sea = data_sea.sum(axis=1).sort_values(ascending=False)
top_sea.iplot(kind='bar')


# In[ ]:


data_sea.transpose().iplot(kind='line')


# After seeing the graph, I immediately checked the political of Myanmar in 2006. They experienced a tough time (you can google about it)

# In[ ]:





# # Expenditure per Capital

# I found a dataset about World Population, let calculate the Expenditure per Capital

# In[ ]:


wp = pd.read_csv('../input/world-population/World Population.csv')


# **Clean data**

# In[ ]:


wp.drop(['Series Name','Series Code','2019 [YR2019]'],axis=1,inplace=True)


# I don't need these columns

# In[ ]:


wp.head()


# In[ ]:


wp.tail(10)


# There are some rows with no data

# In[ ]:


wp.drop(wp.index[264:],inplace=True)


# In[ ]:


wp.info()


# all of number are in type of string. Let convert it to number and errors='coerce' is to transform the non-numeric values into NaN

# In[ ]:


for i in wp.columns[2:]:
    wp[i] = pd.to_numeric(wp[i], errors='coerce')


# In[ ]:


wp.info()


# I think it's ok now.
# <p> Let rename the columns 1 2 to be the same as columns in dataframe ME </p>

# In[ ]:


wp.rename({'Country Name':'Name','Country Code':'Code'},axis=1,inplace=True)


# Now, I will try to devide me to wp to get per Capita. But I need to check the order of data. I suppost I will order by Name

# In[ ]:


def checkCol(df1,df2,colname):
    diff=[]
    c1= df1[colname].tolist()
    c2= df2[colname].tolist()
    for i in range(len(c1)):
        if c1[i] not in c2:
            diff.append(c1[i])
    return diff


# In[ ]:


checkCol(me,wp,'Name')


# Well, there are some differences in Name between 2 dataset, let check the code

# In[ ]:


checkCol(me,wp,'Code')


# Wonderful, so I will order by Country Code:

# In[ ]:


t1 = me.sort_values('Code').drop(['Name','Code','Type','Indicator Name'],axis=1)
t2 = wp.sort_values('Code').drop(['Name','Code'],axis=1)
t2.columns = t1.columns


# In[ ]:


perCapita = t1/t2


# In[ ]:


perCapita = pd.concat([me[['Name','Code','Type']],perCapita],axis=1)


# In[ ]:


perCapita.fillna(value=0,inplace=True)


# I got the perCapita now. Let visualize it:

# In[ ]:


dperCapital = perCapita[perCapita['Type'].apply(lambda x: x in ['Country'])] .drop(['Code','Type'],axis=1).set_index('Name')
top10_perCapital = dperCapital.sum(axis=1).sort_values(ascending=False)[:10]
data_index = top10_perCapital.index
data_top10_perCapital = dperCapital.loc[data_index.tolist()]


# In[ ]:


plt.figure(figsize=(14,8))
sns.heatmap(data_top10_perCapital)


# In[ ]:


data_top10_perCapital.transpose().iplot(kind='line')


# Do you notice any abnormal things?

# # Visualize to World map

# I choose to show the most recent numbers (in 2018) to Map

# In[ ]:


x = me[me['Type'].apply(lambda x: x in ['Country'])][['Name','Code','Type','2018']]


# In[ ]:


y = perCapita[perCapita['Type'].apply(lambda x: x in ['Country'])][['2018']]


# In[ ]:


y.columns=['per Capital 2018']


# In[ ]:


datamap = pd.concat([x, y],axis=1)


# In[ ]:


datamap['per Capital 2018'] = round(datamap['per Capital 2018'],2).astype(str)


# In[ ]:


datamap['Text'] = datamap['Name'] + '<br>' +     'per Capital 2018: $' + datamap['per Capital 2018']


# In[ ]:


datamap = datamap[datamap['Code'].apply(lambda x: x not in ['USA','CHN'])]


# I remove USA and China because they have a really large of number and will affect the color scale. Better to remove them

# In[ ]:


data = dict (type='choropleth',
            locations= datamap['Code'],
            z = datamap['2018'],
            text = datamap['Text'],
            marker_line_color='darkgray',
            marker_line_width=0.5,
            colorbar_tickprefix = '$',
            colorbar = {'title' : 'Military Expenditure Total'}
            )

layout = dict (title = '2018 Global Military Expenditure',
               geo = dict (showframe = True,
                           scope='world',
                          projection = {'type':'equirectangular'}),
              )


# In[ ]:


choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)


# Thank you so much. I will improve this later
