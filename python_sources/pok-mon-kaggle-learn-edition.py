#!/usr/bin/env python
# coding: utf-8

# ALL these is learnt from **KAGGLE LEARN - VISUALIZATION**: https://www.kaggle.com/learn/data-visualization
# This is for practice only, please refer to it for more infos

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #data visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


pokemon = pd.read_csv("../input/pokemon.csv", index_col=0)
pokemon.head()


# **PANDA**

# In[ ]:


#Univariate Plot
pokemon.type1.value_counts().plot.bar() #for categorical data with small numbers - easy to display


# In[ ]:


pokemon.hp.value_counts().sort_index().plot.line() #for continuous values (or points) or large amount of categorical data 
                                                    #easy to show relationship and display large amount of categorical data


# In[ ]:


pokemon.hp.value_counts().sort_index().plot.area() #just line chart in area
#if no sort_index() - then the datas will be very messy


# In[ ]:


pokemon.weight_kg.plot.hist()


# In[ ]:


pokemon.plot.scatter(x = 'sp_attack', y ='sp_defense') #plot correlation


# In[ ]:


pokemon.plot.hexbin(x= 'sp_attack', y = 'sp_defense', gridsize = 15) #show the datas concentrated point


# In[ ]:


pokemon_stats_legendary = pokemon.groupby(['is_legendary', 'generation']).mean()[['sp_attack', 'sp_defense']]
pokemon_stats_legendary.plot.bar(stacked = True)


# In[ ]:


pokemon_stats_by_generation = pokemon.groupby('generation').mean()[['hp', 'weight_kg', 'height_m', 'sp_attack', 'sp_defense', 'speed']]
pokemon_stats_by_generation.plot.line(stacked =True)  # show the diff in generation for all stats comparison of their means
#such as hp/weight_kg/height_m/sp_attack/sp_defense/speed


# **SEABORN**

# In[ ]:


sns.countplot(pokemon.generation) #bar chart in seaborn (no value_count needed as pandas)


# In[ ]:


sns.distplot(pokemon.hp) #where kdeplot only display line - but distplot show bar chart and kdeplot


# In[ ]:


sns.jointplot(x='sp_attack', y='sp_defense', data=pokemon) #same as scatter plot in pandas
#with extra distribution bar chart on top and right


# In[ ]:


sns.jointplot(x='sp_attack', y='sp_defense', data=pokemon, kind = 'hex', gridsize = 20) 
# scatterplot with hex mode on!


# In[ ]:


sns.kdeplot(pokemon.hp,pokemon.sp_attack)
#similar as graph above just better presentation
#high computational cost,and no need to specify x and y, just name of data columns then done


# In[ ]:


sns.boxplot(x = pokemon.is_legendary, y = pokemon.sp_attack)
#or can state with x='islegendary', y = ... , data = pokemon (same result)
#use to show the relative distribution comparison of 2 datas and find 75% , 50% and 25% percentile of datas


# In[ ]:


sns.violinplot(x = pokemon.is_legendary, y = pokemon.sp_attack) #another way of boxplot


# Check more beautiful data presentation with https://seaborn.pydata.org/examples/index.html 

# **FacetGrid + Pairplot with seaborn**

# In[ ]:


legendary_facet = sns.FacetGrid(data = pokemon, row = 'is_legendary')
legendary_facet.map(sns.kdeplot,'sp_attack') #used to plot several graph


# In[ ]:


legendary_facet = sns.FacetGrid(data = pokemon, row = 'generation',col = 'is_legendary')
legendary_facet.map(sns.kdeplot,'sp_attack') #used to plot several graph and arrange in order


# In[ ]:


stats = ['hp','sp_attack','sp_defense']
sns.pairplot(pokemon[stats]) #used to show first insight of the correlation of the datas


# In[ ]:


**Multivariate plot with seaborn**


# **Multivariate plot with seaborn**

# In[ ]:


sns.lmplot(x = 'sp_attack', y = 'sp_defense', markers = ['o','x'],
           hue = 'is_legendary',
           data = pokemon, fit_reg = False) #o x scatter plot


# In[ ]:


sns.boxplot(x='generation',y='base_total', hue='is_legendary',
           data = pokemon)


# In[ ]:


p = (pokemon.loc[:,['hp','sp_attack','sp_defense','attack','defense','speed']]).corr()
sns.heatmap(p,annot = True)


# In[ ]:


from pandas.plotting import parallel_coordinates

p = (pokemon[(pokemon['type1'].isin(["psychic", "fighting"]))]
         .loc[:, ['type1', 'attack', 'sp_attack', 'defense', 'sp_defense']] 
     # .loc need to state all rows, and the columns needed in 2 []
    )

parallel_coordinates(p, 'type1') #datas and the x-label


# **Additional: Plotly (high level API)**

# In[ ]:


from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

iplot([go.Scatter(x = pokemon.attack, y = pokemon.defense, mode = 'markers')]) #Interactive scatterplot


# In[ ]:


iplot([go.Histogram2dContour(x=pokemon.attack, 
                             y=pokemon.defense, 
                             contours=go.Contours(coloring='heatmap')), #Interactive heatmap on plotly
       go.Scatter(x=pokemon.sp_attack, y=pokemon.sp_defense, mode='markers')]) #Interactive scatter on plotly
#combine two interactive map together


# In[ ]:


df = pokemon.assign(n=0).groupby(['attack', 'defense'])['n'].count().reset_index()
pokemon_z = df.pivot(index='attack', columns='defense', values='n').fillna(0).values.tolist() #3D chart
iplot([go.Surface(z=pokemon_z)]) #not sure how to explain


# In[ ]:


#df = reviews['country'].replace("US", "United States").value_counts()

#iplot([go.Choropleth(
#    locationmode='country names',
#    locations=df.index.values,
#    text=df.index,
#    z=df.values
#)])
#this display world map as values distribution and interactive


# For more documentation on interactive plot - **plotly offline** = https://plot.ly/python/

# **PLOTNINE method for graph plotting** 

# In[ ]:


#A really simple plotting method : 1st : U declare the data by using - (ggplot(data_name)
# 2nd: U declare the x and y of the datas: + aes(x= ___,y=___)
# 3rd: U declare the display of your plot: Examples are: point chart = + geom_point()
#smooth line = + stat_smooth()
#color point = + aes(color = 'x/ y')
#facet = + facet_wrap('~ x/y')
#bar chart = + geom_bar()
#hexbin = + geom_bin2d(bins=20) )
#and many more can be config yourself


# In[ ]:


from plotnine import * #import plotnine func

(ggplot(pokemon)
 +aes(x='attack',y='defense')
 +geom_point()) #normal scatter graph


# In[ ]:


(
ggplot(pokemon)
 +aes(x='attack',y='defense',color = 'is_legendary')
 +geom_point()
 +ggtitle("Pokemon Attack and Defense by Legendary Status")
)


# In[ ]:


( 
 ggplot(pokemon)
 +aes(x='attack')
 +facet_wrap('generation')
 +geom_histogram()
)


# Hence, for more info about customization of plotnine can checkout https://plotnine.readthedocs.io/en/stable/api.html (official API of plotnine)

# Special note for last chapter of kaggle learn -> https://www.kaggle.com/residentmario/time-series-plotting-optional
# where time-series plotting is shown, which can be greatly use to visualize the datas in stock market, but pokemon cant be used here ( no time bruh )
# .
# .
# .
# If you like the graph plotting please remember to support kaggle learn data visualization for more info, and upvote this kernel **Thanks :D**
