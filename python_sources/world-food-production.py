#!/usr/bin/env python
# coding: utf-8

# ## Analyzing the Food Production around the Globe.

# 
# <img src="data:image/png;base64,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" width="400px">

# In[ ]:


# for basic operations
import numpy as np
import pandas as pd

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# for advanced visualizations
import folium
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# for providing path
import os
print(os.listdir('../input/'))


# In[ ]:



# reading the data
data = pd.read_csv('../input/FAO.csv', encoding = "ISO-8859-1")

# ENODING ISO-8859-1 is a single byte encoding which can represent the first 256 unicode characters
# Both UTF-8 and ISO-8859-1 encode the ASCII Characters the same.

# checking the shape of the data
print(data.shape)


# In[ ]:


data.head()


# In[ ]:


# adding a total production column

data['total'] = (data['Y1961'] + data['Y1962'] + data['Y1963'] + data['Y1964'] + data['Y1965'] + data['Y1966'] + 
    data['Y1967'] + data['Y1968'] + data['Y1969'] + data['Y1970'] + data['Y1971'] + data['Y1972'] + data['Y1973'] +
    data['Y1974'] + data['Y1975'] + data['Y1976'] + data['Y1977'] + data['Y1978'] + data['Y1979'] + data['Y1980'] + 
    data['Y1981'] + data['Y1982'] + data['Y1983'] + data['Y1984'] + data['Y1985'] + data['Y1986'] + data['Y1987'] + 
    data['Y1988'] + data['Y1989'] + data['Y1990'] + data['Y1991'] + data['Y1992'] + data['Y1993'] + data['Y1994'] + 
    data['Y1995'] + data['Y1996'] + data['Y1997'] + data['Y1998'] + data['Y1999'] + data['Y2000'] + data['Y2001'] + 
    data['Y2001'] + data['Y2002'] + data['Y2003'] + data['Y2004'] + data['Y2005'] + data['Y2006'] + data['Y2007'] + 
    data['Y2008'] + data['Y2009'] + data['Y2010'] + data['Y2011'] + data['Y2012'] + data['Y2013'] )


# In[ ]:


data.describe()


# ## Data Visualization

# In[ ]:


df = data['Area'].value_counts().sort_index().index
df2 = data.groupby('Area')['total'].agg('mean')

trace = go.Choropleth(
    locationmode = 'country names',
    locations = df,
    text = df,
    colorscale = 'Picnic',
    z = df2.values
)
df3 = [trace]
layout = go.Layout(
    title = 'Mean Production in Differet Parts of World')

fig = go.Figure(data = df3, layout = layout)
iplot(fig)


# In[ ]:


df = data['Area'].value_counts().sort_index().index
df2 = data.groupby('Area')['Y1961'].agg('mean')

trace = go.Choropleth(
    locationmode = 'country names',
    locations = df,
    text = df,
    colorscale = 'Rainbow',
    z = df2.values
)
df3 = [trace]
layout = go.Layout(
    title = 'Mean Production in 1961 in Differet Parts of World')

fig = go.Figure(data = df3, layout = layout)
iplot(fig)


# In[ ]:


df = data['Area'].value_counts().sort_index().index
df2 = data.groupby('Area')['Y2013'].agg('mean')

trace = go.Choropleth(
    locationmode = 'country names',
    locations = df,
    text = df,
    colorscale = 'Hot',
    z = df2.values
)
df3 = [trace]
layout = go.Layout(
    title = 'Mean Production in 2013 in Differet Parts of World')

fig = go.Figure(data = df3, layout = layout)
iplot(fig)


# In[ ]:


# delete the total column

data = data.drop(['total'], axis = 1)


# In[ ]:


color = plt.cm.Wistia(np.linspace(0, 1, 40))
plt.style.use('_classic_test')

data['Area'].value_counts().sort_values(ascending = False).head(40).plot.bar(figsize = (20, 10), color = color)
plt.title('Number of Different Items produced by Different Countries in the World', fontsize = 20)
plt.xlabel('Name of the Countries', fontsize = 10)
plt.show()


# In[ ]:


# Top Products around the globe

# setting the style to be ggplot
plt.style.use("dark_background")

items = pd.DataFrame(data.groupby("Item")["Element"].agg("count").sort_values(ascending=False))[:100]

# plotting
plt.rcParams['figure.figsize'] = (15, 20)
#plt.gcf().subplots_adjust(left = .3)
sns.barplot(x = items.Element, y = items.index, data = items, palette = 'Reds')
plt.gca().set_title("Top 100 items produced around globe", fontsize = 30)
plt.show()


# In[ ]:


# setting the size of the plot
plt.rcParams['figure.figsize'] = (20, 20)


# looking at India's Growth
india_production = pd.DataFrame(data[data['Area'] == 'India'].loc[:, "Y2003": "Y2013"].agg("sum", axis = 0))

india_production.columns = ['Production']
plt.subplot(231)
sns.barplot(x = india_production.index, y = india_production.Production, data = india_production, palette = 'PuBu')
plt.gca().set_title("India's Growth")

# looking at china's growth
china_production = pd.DataFrame(data[data['Area'] == 'China, mainland'].loc[:, "Y2003":"Y2013"].agg("sum", axis = 0))

china_production.columns = ['Production']
plt.subplot(232)
sns.barplot(x = china_production.index, y = india_production.Production, data = china_production, palette = 'RdPu')
plt.gca().set_title("China's Growth")

#looking at usa's growth
usa_production = pd.DataFrame(data[data['Area'] == 'United States of America'].loc[:,"Y2003":"Y2013"].agg("sum", axis = 0))

usa_production.columns = ['Production']
plt.subplot(233)
sns.barplot(x = usa_production.index, y = usa_production.Production, data = usa_production, palette = 'Blues')
plt.gca().set_title("USA's Growth")

#looking at brazil's growth
brazil_production = pd.DataFrame(data[data['Area'] == 'Brazil'].loc[:,"Y2003":"Y2013"].agg("sum", axis = 0))

brazil_production.columns = ['Production']
plt.subplot(234)
sns.barplot(x = brazil_production.index, y = brazil_production.Production, data = brazil_production, palette = 'Purples')
plt.gca().set_title("Brazil's Growth")


#looking at mexico's growth
mexico_production = pd.DataFrame(data[data['Area'] == 'Mexico'].loc[:,"Y2003":"Y2013"].agg("sum", axis = 0))

mexico_production.columns = ['Production']
plt.subplot(235)
sns.barplot(x = mexico_production.index, y = mexico_production.Production, data = mexico_production, palette = 'ocean')
plt.gca().set_title("Mexico's Growth")

#looking at russia's growth
russia_production = pd.DataFrame(data[data['Area'] == 'Russian Federation'].loc[:,"Y2003":"Y2013"].agg("sum", axis = 0))

russia_production.columns = ['Production']
plt.subplot(236)
sns.barplot(x = russia_production.index, y = russia_production.Production, data = russia_production, palette = 'spring')
plt.gca().set_title("Russia's Growth")

plt.suptitle('Top 6 Countries Growth from 2003 to 2013', fontsize = 30)
plt.show()


# In[ ]:


labels = ['Feed', 'Food']
size = data['Element'].value_counts()
colors = ['cyan', 'magenta']
explode = [0.1, 0.1]

plt.rcParams['figure.figsize'] = (10, 10)
plt.pie(size, labels = labels, colors = colors, explode = explode, shadow = True)
plt.axis('off')
plt.title('A Pie Chart Representing Types of Elements', fontsize = 20)
plt.legend()
plt.show()


# In[ ]:


# plotting for the Annual Production of crops by every country

countries = list(data['Area'].unique())
years = list(data.iloc[:, 10:].columns)

plt.style.use('seaborn')    
plt.figure(figsize = (20, 20))
for i in countries:
    production = []
    for j in years:
        production.append(data[j][data['Area'] == i].sum())
    plt.plot(production, label = i)
    
plt.xticks(np.arange(53), tuple(years), rotation = 90)
plt.title('Country wise Annual Production')
plt.legend()
plt.legend(bbox_to_anchor = (0., 1, 1.5,  1.5), loc = 3, ncol = 12)
plt.savefig('p.png')
plt.show()


# In[ ]:


# creating a new data containing information about countries and productions only

new_data_dict = {}
for i in countries:
    production = []
    for j in years:
        production.append(data[j][data['Area'] == i].sum())
    new_data_dict[i] = production
new_data = pd.DataFrame(new_data_dict)

new_data.head()


# In[ ]:


new_data['Year'] = np.linspace(1961, 2013, num = 53).astype('int')

# checking the shape of the new data
new_data.shape


# In[ ]:



#heatmap

plt.rcParams['figure.figsize'] = (15, 15)
plt.style.use('fivethirtyeight')
 
sns.heatmap(new_data, cmap = 'PuBu')
plt.title('Heatmap for Production', fontsize = 20)
plt.yticks()
plt.show()


# From the above graph we can see that the top producers are India, China, USA, Russia, and Brazil

# ## Time Series Analysis for Top Producers

# In[ ]:


plt.rcParams['figure.figsize'] = (15, 8)
plt.style.use('dark_background')

sns.lineplot(new_data['Year'], new_data['United States of America'], color = 'yellow')
plt.title('Time Series Analysis for USA', fontsize = 30)
plt.grid()
plt.show()


# In[ ]:


plt.rcParams['figure.figsize'] = (15, 8)
plt.style.use('dark_background')

sns.lineplot(new_data['Year'], new_data['India'], color = 'yellow')
plt.title('Time Series Analysis for India', fontsize = 30)
plt.grid()
plt.show()


# In[ ]:


plt.rcParams['figure.figsize'] = (15, 8)
plt.style.use('dark_background')

sns.lineplot(new_data['Year'], new_data['China, mainland'], color = 'yellow')
plt.title('Time Series Analysis for China', fontsize = 30)
plt.grid()
plt.show()


# In[ ]:


plt.rcParams['figure.figsize'] = (15, 8)
plt.style.use('dark_background')

sns.lineplot(new_data['Year'], new_data['Russian Federation'], color = 'yellow')
plt.title('Time Series Analysis for Russia', fontsize = 30)
plt.grid()
plt.show()


# In[ ]:


plt.rcParams['figure.figsize'] = (15, 8)
plt.style.use('dark_background')

sns.lineplot(new_data['Year'], new_data['Iceland'], color = 'yellow')
plt.title('Time Series Analysis for Iceland', fontsize = 30)
plt.grid()
plt.show()


# In[ ]:


plt.rcParams['figure.figsize'] = (15, 8)
plt.style.use('dark_background')

sns.lineplot(new_data['Year'], new_data['Brazil'], color = 'yellow')
plt.title('Time Series Analysis for Brazil', fontsize = 30)
plt.grid()
plt.show()


# 
# <img src="https://media.giphy.com/media/3oz8xK0liGg1yTgcZq/giphy.gif" width="400px">
