#!/usr/bin/env python
# coding: utf-8

# **<i><font size="4" color='#3F1F0F'>Welcome On Board Craft Beer Lovers :) </font></i>**
# 
# ![](https://static1.squarespace.com/static/573b564945bf21707e4a1108/573b5b5c60b5e90d555b40e2/576b4120440243decaa2db34/1466647077750/033BWO-7295a.jpg)

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

import matplotlib.pyplot as plt 
import seaborn as sns

from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

import codecs

from PIL import Image
from termcolor import colored

#to supress Warnings 
import warnings
warnings.filterwarnings("ignore")


#plotly
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
import plotly.graph_objs as py

# Any results you write to the current directory are saved as output.


# In[ ]:


#using all the required files from the datasets 
beer = pd.read_csv("../input/craft-cans/beers.csv")
brewery = pd.read_csv("../input/craft-cans/breweries.csv")
with codecs.open("../input/beer-recipes/recipeData.csv", "r",encoding='utf-8', errors='ignore') as fdata:
    brrecipe = pd.read_csv(fdata)
GlobalBeer = pd.read_csv("../input/beer-beer-beer/open-beer-database.csv")


# In[ ]:


beer.describe()
beer.head(10)
brewery.describe()
brewery.head(10)


# ***<font size="5" color="#D4AF37">Popular Beers</font>***

# In[ ]:


print(colored(beer.groupby('name')["name"].count().sort_values(ascending=False).head(10),'green'))


# In[ ]:


#word cloud - Visualizing  Craft Beers
wave_mask = np.array(Image.open( "../input/beerimage/images.jpg")) 

wordcloud = (WordCloud(width=1440, height=1080, relative_scaling=0.5, mask=wave_mask,background_color='white').generate_from_frequencies(beer['name'].value_counts()))

fig = plt.figure(1,figsize=(15, 15))
plt.imshow(wordcloud,interpolation="hanning")
plt.axis('off')
plt.show()


# ***<font size="5" color="#D4AF37">Popular Breweries in United States</font>***

# In[ ]:


print(colored(brewery.groupby('name')["name"].count().sort_values(ascending=False).head(10),'blue'))


# In[ ]:


#word cloud - Visualizing  Breweries  
mask_glass = np.array(Image.open( "../input/beerimage/english-brown-porter.jpg"))
wordcloud = (WordCloud(width=1440, height=1080, mask = mask_glass, relative_scaling=0.5, background_color='#fbf7ed',stopwords=stopwords).generate_from_frequencies(brewery['name'].value_counts()))

fig = plt.figure(1,figsize=(15, 15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# ***<font size="5" color="#D4AF37">Popular Beer Styles </font>***

# In[ ]:


print(colored(beer.groupby('style')["name"].count().sort_values(ascending=False).head(10),'green'))


# In[ ]:


#word cloud - Visualizing  Beer Style  
wave_masknew = np.array(Image.open( "../input/beerimage/carlsberg-beer-650ml-mynepshopcom.jpg"))
wordcloud = (WordCloud( max_font_size=100, min_font_size=8,mask = wave_masknew, stopwords=stopwords,background_color='#E6BE8A').generate(str(beer['style'])))
fig = plt.figure(1,figsize=(15, 15))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


# ***<font size="5" color="#D4AF37">Cities for Beer Lovers</font>***

# In[ ]:


print(colored(brewery.groupby('city')["city"].count().sort_values(ascending=False).head(10),'blue'))


# In[ ]:


#word cloud - Popular Cities (top 50)
wave_mask3 = np.array(Image.open( "../input/beerimage/tulip-beer-glass.jpg"))
wordcloud = (WordCloud( max_words = 200,stopwords=stopwords, mask = wave_mask3, background_color='#361F1B',margin=10).generate_from_frequencies(brewery['city'].value_counts()))
fig = plt.figure(1,figsize=(15, 15))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


# **<font size="5" color="#D4AF37">Top 3 States for Beer Lovers </font>**

# In[ ]:


brewery.groupby('state')['state'].count().sort_values(ascending=False).head(3)
print(colored("Top 3 States with Maximum Breweries for Craft Canned Beers:",'blue'))
#Manually converting state initials to names
print(colored("1. Colorado",'green'))
print(colored("2. California",'green'))
print(colored("3. Michigan",'green'))


# **<font size="5" color="#D4AF37">Number of Beers by Alcohol Content </font><font size="5" color="green">(top 50)</font>**

# In[ ]:


temp = beer.groupby('abv')["name"].count().sort_values(ascending=False).head(50)

#Craft Beers Alcohol content
x = list(temp.index.values)
for i in range(len(x)):
    x[i] = np.format_float_positional(np.float16(x[i]*100))
y = temp.values

fig_size = plt.rcParams["figure.figsize"]
# Set figure width to 12 and height to 9
fig_size[0] = 27
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size
sns.barplot(x,y)
plt.xlabel("Alcohol Volume (%)",color='green')
plt.ylabel("Number of Beers",color='green')
plt.title("Craft Beer by Alcohol content", color='blue')
plt.show()
#np.format_float_positional(np.float16(np.pi))


# **<font size="5" color="#D4AF37"><i>Bitterness</i> vs <i>Alcohol By Volume</i> of Craft Beers</font>**

# In[ ]:


beertemp = beer
beertemp.dropna(inplace=True)
sns.lmplot("ibu","abv", data=beertemp,markers="x")
plt.xlabel("Bitterness (ibu)",color='blue')
plt.ylabel("Absolute Volume (abv) - Alcohol Content",color='blue')
plt.title("Craft Beers -- Bitterness vs Alcohol Content ", color='#BE823A')


# **<font size="5" color="#D4AF37">"Does some Beer are highly Bitter & some have high Alcohol Content?"</font>**

# In[ ]:


fig = plt.figure()

plt.subplot(1, 2, 1)
sns.boxplot(beertemp["ibu"],color='#361F1B')
plt.xlabel("International Bitterness Units (ibu)")

plt.subplot(1, 2, 2)
sns.boxplot(beertemp["abv"],color='#fbf7ed')
plt.xlabel("Alcohol By Volume")

plt.show()


# **<font size="5" color="#D4AF37">Can I get a Craft Beer with high Alcohol Content and Average Bitterness?</font>**

# In[ ]:


HighBeer = beertemp.sort_values("abv",ascending=False)
AvgBitterness = beertemp["ibu"].mean()
HighBeer = HighBeer[HighBeer["ibu"]<=AvgBitterness].reset_index().head(1)
HighBeer[["name","style"]]


# **<font size="5" color="#D4AF37">How do they Brew It!</font><font size="3" color="green"> Brew Method by IBU & ABV Plot</font>**

# In[ ]:


#Most Popular Brew Method -- using the New Dataset of Beer Recipes
brtemp["BrewMethod"].value_counts()
#Its description
brrecipe[brrecipe["BrewMethod"]=="All Grain"].describe()


# In[ ]:


brtemp = brrecipe.fillna(0)
brtemp["IBU"]=brtemp["IBU"]/10
plt.figure(figsize=(30 ,30))
sns.lmplot( x="ABV",y="IBU",  data=brtemp, fit_reg=False, hue='BrewMethod', legend=False)
plt.legend(loc='upper left')

plt.show()


# **<font size="4" color="#D4AF37"> How does different factors in Beer Production correlate with each other!</font>**

# In[ ]:


#how does different factor in Beer Production correlate with each other 
# Compute the correlation matrix
corr = brrecipe.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# **<font size="5" color="#D4AF37">Look at Some Major Factors in Beer Preparation <font size="5" color="green">using Seaborn Pair-Plot</font>**

# In[ ]:


#Data Prep for PairPlot
brr = brrecipe.iloc[:,5:10]
brr["Boil Time"] = brrecipe["BoilTime"]
brr = brr.rename(columns={"OG": "Original Gravity", "FG": "Final Gravity", "ABV":"Alcohol By Volume", "IBU":"Bitterness (IBU)"})
brr.fillna(0,inplace=True)

#plot Prep 
sns.set(style="ticks", color_codes=True)
sns.pairplot(brr)


# **<font size="5" color="#D4AF37"><i>Popular Beer Countries</i></font>**

# In[ ]:


#word cloud - Countries for Beer
wordcloudC = (WordCloud( background_color='black').generate_from_frequencies(GlobalBeer['Country'].value_counts()))

fig = plt.figure(1,figsize=(15, 15))
plt.imshow(wordcloudC,interpolation="bilinear")
plt.axis('off')
plt.show()


# In[ ]:


#DataPrep
bC = GlobalBeer
bC = bC.dropna(subset=['Coordinates', 'Country'])
coords = bC["Coordinates"]
LongLat = coords.str.split(',').values.tolist()
bC["Latitude"] = pd.Series()
bC["Longitude"] = pd.Series()

j = 0 
for i in LongLat:
    bC['Latitude'][j] = i[0]
    bC['Longitude'][j] = i[1]
    j = j + 1 


# In[ ]:


data1 = [dict(
    type='scattergeo',
    lon = bC['Longitude'],
    lat = bC['Latitude'],
    text = bC['Name'],
    mode = 'markers',
    marker = dict(
    color = '#aa7700',
    )
    
)]
layout = dict(
    title = 'Popular Beer Countries',titlefont=dict(color='#887700'), 
    hovermode='closest',
    geo = dict(showframe=False, countrywidth=1, showcountries=True,showocean=True,showland=True,countrycolor='green', 
               showcoastlines=True, projection=dict(type='natural earth')),
    
)
fig = py.Figure(data=data1, layout=layout)
iplot(fig)


# **<font size="5" color="#D4AF37"><i>Distribution of Alcohol Content in Beers'</i></font>**

# In[ ]:


sns.distplot(brrecipe["ABV"])
plt.xlabel("Alcohol By Volume")
plt.show()


# ***It is a nice Normal Distribution (a single Bell Curve)***

# In[ ]:




