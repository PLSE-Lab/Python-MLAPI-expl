#!/usr/bin/env python
# coding: utf-8

# Things i wat to test on
# 
# Visuallization:
# 1. barplot \ horizontal barplot
# 2. Histogram
# 3. pie chart
# 4. Scatter plot
# 5. boxplot
# 6. violin plot
# 7. heat map
# 8. line graph
# 9. multi bar barplot?
# 10. world map (advanced)
# 11. gif (really advanced)
# 
# pandas:
# 1. simple mean,median,mode
# 2. loc \ iloc to find specific
# 3. groupby
# 4. merge
# 5. Filtering
# 6. sorting
# 7. Maipulating the dataset
# 8. Working with series

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import io
import numpy as np
import pandas as pd
from IPython.display import Image
from matplotlib import pyplot as plt
import seaborn as sns


get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.mode.chained_assignment = None  # default='warn'


# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/globalterror/GlobalTerror.csv')


# In[ ]:


df.columns


# In[ ]:


df.head()


# Show the mean number of casualities from a terror attacl

# In[ ]:


# Level = Easy
# Tests = mean
df['Casualities'].mean()


# Show a Barplot/Countplot of the number of of terrorist attacks every year

# In[ ]:


# Level: Easy / Medium
# Tests: Barplot
plt.subplots(figsize=(15,6))
sns.countplot('Year',data=df,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90) #make x ticks (years) horizontal' so they won't collapse with eachother
plt.title('Number Of Terrorist Activities Each Year')
plt.show()


# Create a horizontal barplot showing the most popular Target Type From Terror Attacks in the North American Region

# In[ ]:


# Level: Easy/Medium
# Tests: Horizontal Barplot, Filtering
plt.subplots(figsize=(15,6))
sns.countplot(y='TargetType',data=df[df['Region'] == 'North America'],orient ='horizontal',palette='inferno')
plt.title('Most common targets in the north american region')
plt.show()


# Show the most used Weapon Types in a piechart

# In[ ]:


# Level: Easy
# Tests: Piechart
df['WeaponType'].value_counts().plot.pie()


# Wow that pie chart is clutterd!, Show only the top 5 values from the piechart, and group the other values into an "other" section

# In[ ]:


# Level: Medium
# Tests: Piechart, Working a series
weapon_series = df['WeaponType'].value_counts()
values_to_show = 5
weapon_series = weapon_series.nlargest(values_to_show).append(pd.Series(weapon_series.nsmallest(weapon_series.size - values_to_show).sum(),index=['other']))
weapon_series.plot.pie()


# Show on a scatter plot ontop of the world map of terror attacks in two colors, blue when the number of casualities is below 75 and red when the number of casualities is above 75

# In[ ]:


# Level: Really Hard
# Tests: Filtering, World Map, Filtering
from mpl_toolkits.basemap import Basemap
import matplotlib.patches as mpatches
m3 = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c',lat_0=True,lat_1=True)
lat_100=list(df[df['Casualities']>=75].latitude)
long_100=list(df[df['Casualities']>=75].longitude)
x_100,y_100=m3(long_100,lat_100)
m3.plot(x_100, y_100,'go',markersize=5,color = 'r')
lat_=list(df[df['Casualities']<75].latitude)
long_=list(df[df['Casualities']<75].longitude)
x_,y_=m3(long_,lat_)
m3.plot(x_, y_,'go',markersize=2,color = 'b',alpha=0.4)
m3.drawcoastlines()
m3.drawcountries()
m3.fillcontinents(lake_color='aqua')
m3.drawmapboundary(fill_color='aqua')
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.title('Global Terrorist Attacks')
plt.legend(loc='lower left',handles=[mpatches.Patch(color='b', label = "< 75 casualities"),
                    mpatches.Patch(color='red',label='> 75 casualities')])
plt.show()


# Create a Gif of a scatter plot ontop of the map of israel, where each frame represents a year and a the dots represent a terror attack, the bigger the dot the more casualities the terror attack had

# for example here is an example of this gif but instead of israel, its for the united states

# In[ ]:


from IPython.display import HTML
import base64
gif = io.open('/kaggle/input/globalterror/UsaTerror.gif', 'rb').read()
encoded = base64.b64encode(gif)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))


# In[ ]:


# Level: Crazy Hard
# Tests: Filtering, WorldMap, ScatterPlot, Gif
from matplotlib import animation,rc
from IPython.display import display
import warnings
terror_israel=df[df['Country']=='Israel']
fig = plt.figure(figsize = (10,8))
def animate(Year):
    ax = plt.axes()
    ax.clear()
    ax.set_title('Terrorism In Israel '+'\n'+'Year:' +str(Year))
    m6 = Basemap(projection='mill',llcrnrlat=28,llcrnrlon=34,urcrnrlat=35,urcrnrlon=37,lat_ts=20,resolution='c',lat_0=True,lat_1=True)
    lat_gif1=list(terror_israel[terror_israel['Year']==Year].latitude)
    long_gif1=list(terror_israel[terror_israel['Year']==Year].longitude)
    x_gif1,y_gif1=m6(long_gif1,lat_gif1)
    m6.scatter(x_gif1, y_gif1,s=[killed+wounded for killed,wounded in zip(terror_israel[terror_israel['Year']==Year].Killed,terror_israel[terror_israel['Year']==Year].Wounded)],color ='r') 
    m6.drawcoastlines()
    m6.drawcountries()
    m6.fillcontinents(color='white',lake_color='aqua', zorder = 1,alpha=0.4)
    m6.drawmapboundary(fill_color='aqua')
ani = animation.FuncAnimation(fig,animate,list(terror_israel.Year.unique()), interval = 1500)    
ani.save('animation.gif', writer='imagemagick', fps=1)
plt.close(1)
filename = 'animation.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))


# Show The sum of casualities per region (on a barplot)

# In[ ]:


# Level: Easy
# Tests: Groupby, mean, barplot
casualities_per_region = df.groupby('Region')['Casualities'].sum()
sns.barplot(casualities_per_region.index,casualities_per_region.values)
plt.xticks(rotation=90) #make x ticks (years) horizontal' so they won't collapse with eachother
plt.title('Number of casualities per region')
plt.show()


# Find the 9/11 terror attack and print the summary of it (Hint, the "9/11" attack actually refers to multipule attacks which happend in the same time in the United States)

# In[ ]:


# Level: Easy
# Tests: Filtering
list(df[(df['Day'] == 11) & (df['Month'] == 9) & (df['Year'] == 2001) & (df['Country'] == 'United States')]['Summary'])


# Show the summary of the terror attack in index 73905

# In[ ]:


# Level: Easy
# Tests: loc / iloc
df.loc[73905]['Summary']


# Show a boxplot of casualities per group from the top 10 deadliest groups (who had the most casualities summed up)

# In[ ]:


# Level: Medium / Hard
# Tests: Boxplot, Filtering, Groupby
top_n = 10
top_n_killer_groups = list(df.groupby('Group')['Casualities'].sum().sort_values(ascending=False).nlargest(top_n).index)
sns.boxplot(x = "Group", y = "Casualities",data = df[df['Group'].isin(top_n_killer_groups)])
plt.xticks(rotation=90) #make x ticks (years) horizontal' so they won't collapse with eachother
plt.title('Number of casualities per region')
plt.ylim(0,150)
plt.show()


# Find the terror attacks with the most killed, print the country it was in, the group who performed it and the number of killed

# In[ ]:


# Level: Very Easy
# Tests: sorting
most_deadly_attack = df.sort_values(by='Killed',ascending=False).max()
print(f'The most deadly terror attack was preformed in {most_deadly_attack["Country"]} by {most_deadly_attack["Group"]} and {int(most_deadly_attack["Killed"])} were killed')


# Show a lineplot of the number of terror attacks over the year by the top 10 infamous groups (who preformed the most terror attacks overall)

# In[ ]:


# Level: Hard
# Tests: Lineplot, Groupby, Working with series
top_groups10=df['Group'].value_counts()[1:11].index
fig, ax = plt.subplots(figsize=(15,7))
df[df['Group'].isin(top_groups10)].groupby(['Year','Group'])['Group'].value_counts().droplevel(2).unstack().fillna(0).plot(ax=ax)


# Show a Grouped barplot of the total number of Killed and Wounded people in each region

# In[ ]:


# Level: Easy / Hard
# Tests: groupby, Group barplot
killed_and_wounded_per_region = df.groupby('Region')[['Killed','Wounded']].sum()
killed_and_wounded_per_region.plot(kind='barh')


# In[ ]:


# or using seaborn
killed_and_wounded_per_region = df.groupby('Region')[['Killed','Wounded']].sum()
killed_and_wounded_per_region = killed_and_wounded_per_region.reset_index()
killed_and_wounded_per_region = killed_and_wounded_per_region.melt('Region',var_name='a', value_name='b')
sns.barplot(x='b', y='Region', hue='a', data=killed_and_wounded_per_region)


# Find The most Common attack type in india

# In[ ]:


# Level: Easy
# Tests: Mode, Filtering
df[df['Country'] == 'India']['AttackType'].mode()


# for each Attack type, retrive the country which that attack type happens the most in 
# (In other words, retrive the **mode** country foreach attack type)
# (Hint, read on the SeriesGroupBy.apply)

# In[ ]:


# Level: Hard
# Tests: Groupby, Working with series, mode
mode_attackType_per_country = df.groupby('AttackType')['Country'].apply(lambda x: x.mode())
mode_attackType_per_country


# Based on the previous question, show a lineplot of the Armed Assault attacks in the country which it is most common through the years

# In[ ]:


# Level: Easy
# Tests: Lineplot

fig, ax = plt.subplots(figsize=(15,7))
df[(df['AttackType'] == 'Armed Assault') & (df['Country'] == 'Pakistan')].groupby('Year').size().plot(ax=ax)


# We have another data set, with statistical data about countries

# In[ ]:


country_stats = pd.read_csv('/kaggle/input/undata-country-profiles/country_profile_variables.csv')


# In[ ]:


country_stats.head()


# As you can see the region of this dataframe is formatted differently then the one we have in our terrorist attacks dataset.
# create a new dataset which has 3 columns, the name of the country, the population of the country and the region as it is mentioned in the terrorist attacks dataset (countries which do not appear in our terrorist attacks dataset are not important for us, so they don't have to be included)
# (notice that the population in this dataset is in thousands, we need the number of citizens (not in thousands))

# In[ ]:


# Level: Hard
# Tests, Merge, Maipulating the dataset
merged = country_stats.merge(df,left_on='country',right_on='Country',suffixes=('_left', '_right'),how='right')
merged['Region'] = merged['Region_right']
merged['Population'] = merged['Population in thousands (2017)'] * 1000
merged= merged.groupby(['country'])[['Region','Population']].first().reset_index()

merged


# Create a scatter plot in which every dot represents a country. the x axis represents the number of attacks per citizen and the y axis represents the number of killed citizens per citizen. make each region be in a different color

# In[ ]:


# Level: Hard
# Tests: Scatterplot, working with series
countries_with_population = df[df['Country'].isin(merged['country'])]
attacks_per_citizen = countries_with_population['Country'].value_counts().sort_index() / merged.set_index('country')['Population']
killed_per_citizen = countries_with_population.groupby('Country')['Killed'].sum() / merged.set_index('country')['Population']
merged.set_index('country')['Region']
plot_df = pd.concat([attacks_per_citizen,killed_per_citizen,merged.set_index('country')['Region']], axis=1, keys=['attacks', 'killed','region'])


plt.subplots(figsize=(15,6))
sns.scatterplot(x='attacks', y='killed',data=plot_df, hue='region')

