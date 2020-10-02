#!/usr/bin/env python
# coding: utf-8

# <center><img src='https://developer.apple.com/news/images/og/app-store-og.png' width=850></center>

# <div align='left'><font size="6" color="#ff3fb5" id='sec1'>Table Of Contents</font></div>
# >* [Data Description](#sec1)
# >* [Exploring Data](#sec2)
# >* [Different Genres](#sec3)
# >* [Evolution of Games Over the Time](#sec4)
# >* [Are Paid Games Really Good ?](#sec5)
# >* [Age Ratings?](#sec6)
# >* [Most Expensive Game](#sec7)
# >* [Most Reviewed/Popular Game](#sec8)
# >* [Best Overall Game](#sec9)

# In[ ]:


get_ipython().system('pip install bubbly')


# In[ ]:


import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px
py.init_notebook_mode(connected=True)

from bokeh.io import output_notebook, output_file, show, push_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CategoricalColorMapper, HoverTool
from bokeh.models.widgets import Tabs, Panel
output_notebook()

from IPython.html.widgets import interact

from bubbly.bubbly import bubbleplot
get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print('Data:-')
        print(os.path.join(dirname, filename))


# <div align='center'><font size="6" color="#ff3fb5" id='sec1'>Data Description</font></div>

# * URL:-The URL<br>
# * ID:-The assigned ID<br>
# * Name:-The name<br>
# * Subtitle:-The secondary text under the name<br>
# * Icon URL:-512px x 512px jpg<br>
# * Average User Rating:-Rounded to nearest .5, requires at least 5 ratings<br>
# * User Rating Count:-Number of ratings internationally, null means it is below 5<br>
# * Price:-Price in USD<br>
# * In-app Purchases:-Prices of available in-app purchases<br>
# * Description:-App description<br>
# * Developer:-App developer<br>
# * Age Rating:-Either 4+, 9+, 12+ or 17+<br>
# * Languages:-ISO2A language codes<br>
# * Size:-Size of the app in bytes<br>
# * Primary Genre:-Main genre<br>
# * Genres:-Genres of the app<br>
# * Original Release Date:-When it was released<br>
# * Current Version Release Date:-When it was last updated<br>

# In[ ]:


df = pd.read_csv('../input/17k-apple-app-store-strategy-games/appstore_games.csv', parse_dates=['Original Release Date', 'Current Version Release Date'])
df.head()


# <div align='center'><font size="6" color="#ff3fb5" id='sec2'>Exploring Data</font></div>

# In[ ]:


print(f"There are {df.shape[0]} rows and {df.shape[1]} columns in dataset")


# In[ ]:


df.info()


# > ** Mostly Subtitle and Averge User Ratings and In-app purchases have most null values**

# In[ ]:


df.describe()


# > ** Here we can see mean, min, max values for Average User Rating and User Rating Count, Price, Size**

# In[ ]:


'''Here we are extracting the apps having at least 200 reviews and selecting our primary genre as games'''
df = df.loc[(df['User Rating Count'] > 200) & (df['Primary Genre']=='Games')]


# <div align='center'><font size="6" color="#ff3fb5" id='sec3'>Different Genres ?</font></div>

# In[ ]:


'''A Function To Plot Pie Plot using Plotly'''

def pie_plot(cnt_srs, colors, title):
    labels=cnt_srs.index
    values=cnt_srs.values
    trace = go.Pie(labels=labels, 
                   values=values, 
                   title=title, 
                   hoverinfo='percent+value', 
                   textinfo='percent',
                   textposition='inside',
                   hole=0.7,
                   showlegend=True,
                   marker=dict(colors=colors,
                               line=dict(color='#000000',
                                         width=2),
                              )
                  )
    return trace


# In[ ]:


py.iplot([pie_plot(df['Genres'].value_counts().sort_values(ascending=False).head(10), ['cyan'], 'Genres')])


# > ** What we can figure out is that Games are mostly based on strategy, simulation and action genres in App Store**

# In[ ]:


'''Converting Size in MB's'''
df['Size'] = round(df['Size']/1000000)
fig = bubbleplot(dataset=df, x_column='Average User Rating', y_column='User Rating Count', size_column='Size', bubble_column='Genres',
                 color_column='Genres', x_title='Avg Rating', y_title='Ratings Count', title='Ratings vs Rating_Count', x_logscale=False, 
                 y_logscale=True,
                 scale_bubble=3, height=650)
py.iplot(fig)


# > **Here we can observe and find which genre games are having the highest user rating and more number of reviews and also the smallest/biggest size of game**

# <div align='center'><font size="6" color="#ff3fb5" id='sec4'>Evolution of Games Over the Time</font></div>

# In[ ]:


df['Release Year'] = df['Original Release Date'].dt.year

fig, ax = plt.subplots(1, 2, figsize=(15, 8))
sns.lineplot(x='Release Year', y='Price', data=df, palette='Wistia', ax=ax[0])
ax[0].set_title('Release Year vs Price')

sns.lineplot(x='Release Year', y='Size', data=df, palette='Wistia', ax=ax[1])
ax[1].set_title('Relase Year vs Size')
plt.tight_layout()
plt.show()


# **We Can Observe that prices for the games decreased tremendously but Size of the game increased which is obvious as most of us have access internet and we can easily download a 1-2GB game**

# In[ ]:


df.dropna(inplace=True)
data = df.set_index('Release Year')
y = data.loc[2008].Size
x = data.loc[2008].Price
data = data[['Name', 'Price', 'Size']]
output_notebook()


# <div><font color='#ff3fb5' size='4'>You can observe the price & size ratio for different years</font></div>

# In[ ]:


source = ColumnDataSource(data={
    "x": data.loc[2009].Price,
    "y": data.loc[2009].Size,
    "Name": data.loc[2009].Name,
    "Price": data.loc[2009].Price,
    "Size": data.loc[2009].Size
})
hover = HoverTool(tooltips=[('Name', '@Name'), ('Price', '@Price'), ('Size', '@Size')])
plot = figure(title='Evolution Of Games', x_axis_label='Price', y_axis_label='Size', tools=[hover, 'crosshair', 'pan', 'box_zoom'])
plot.circle('x', 'y' , source=source, hover_color='red')

def update(x_axis, y_axis, year=2009):   
    c1=x_axis
    c2=y_axis
    new_data={
        "x":data.loc[year, c1],
        "y":data.loc[year, c2],
        "Name":data.loc[year].Name,
        "Price": data.loc[year].Price,
        "Size": data.loc[year].Size
     }
    source.data = new_data
    plot.xaxis.axis_label=c1
    plot.yaxis.axis_label=c2
    push_notebook()

show(plot, notebook_handle=True)


# In[ ]:


'''Toggle year from here It will show up when u will fork the kernel and run this cell'''
interact(update, x_axis=['Price'], y_axis=['Size'], year=(2009, 2019))


# <div align='center'><font size="6" color="#ff3fb5" id='sec5'>Are Paid Games Really Good ?</font></div>

# In[ ]:


paid = df[df['Price']>0]
free = df[df['Price']==0]
fig, ax = plt.subplots(1, 2, figsize=(15,8))
sns.countplot(data=paid, y='Average User Rating', ax=ax[0], palette='plasma')
ax[0].set_title('Paid Games')
ax[0].set_xlim([0, 1000])

sns.countplot(data=free, y='Average User Rating', ax=ax[1], palette='viridis')
ax[1].set_title('Free Games')
ax[1].set_xlim([0,1000])
plt.tight_layout();
plt.show()


# >**1. As expected there are less number of paid games than free games**<br>
# **2. But Still we cannot see any difference in user ratings of both the categories**<br>
# **3. Most of the Games are rated quite good around 4.0-5.0 **<br>
# **4. It doesn't seem like price has an impact on the ratings as both free and paid games have almost same ratings**

# <div align='center'><font size="6" color="#ff3fb5" id='sec6'>Age Ratings?</font></div>

# In[ ]:


py.iplot([pie_plot(df['Age Rating'].value_counts(), ['cyan', 'gold', 'red'], 'Age Rating')])


# >** 1. Most of the Games are 4+ and 9+ ** <br>
# ** 2. So Definetly Game Developers are looking for a wide range of audience **

# <div align='center'><font size="6" color="#ff3fb5" id='sec7'>Most Expensive Game ?</font></div>

# In[ ]:


price = df.sort_values(by='Price', ascending=False)[['Name', 'Price', 'Average User Rating', 'Size', 'Icon URL']].head(10)
price.iloc[:, 0:-1]


#  > **SmartGo Kifu and Panzer Corps are the most expensive games on App Store**

# In[ ]:


import urllib.request
from PIL import Image

plt.figure(figsize=(6,3))
plt.subplot(121)
image = Image.open(urllib.request.urlopen(price.iloc[0,-1]))
plt.imshow(image)
plt.axis('off')

plt.subplot(122)
image = Image.open(urllib.request.urlopen(price.iloc[1,-1]))
plt.imshow(image)
plt.axis('off')

plt.show()


# <div align='center'><font size="6" color="#ff3fb5" id='sec8'>Most Reviewed/Popular Game ?</font></div>

# In[ ]:


review = df.sort_values(by='User Rating Count', ascending=False)[['Name', 'Price', 'Average User Rating', 'Size', 'User Rating Count', 'Icon URL']].head(10)
review.iloc[:, 0:-1]


# >**1. Clash Of Clans **<br>
#  **2. Clash Royale **<br>
#  **3. PUBG Mobile ** <br>
# * **Are the Most Reviewed Games and We can also say Popular Games on App Store**

# In[ ]:


plt.figure(figsize=(6,3))
plt.subplot(131)
image = Image.open(urllib.request.urlopen(review.iloc[0,-1]))
plt.imshow(image)
plt.axis('off')

plt.subplot(132)
image = Image.open(urllib.request.urlopen(review.iloc[1,-1]))
plt.imshow(image)
plt.axis('off')

plt.subplot(133)
image = Image.open(urllib.request.urlopen(review.iloc[2,-1]))
plt.imshow(image)
plt.axis('off')

plt.show()


# <div align='center'><font size="6" color="#ff3fb5" id='sec9'>Best Overall Game ?</font></div>

# In[ ]:


best = df.sort_values(by=['Average User Rating', 'User Rating Count'], ascending=False)[['Name', 'Average User Rating', 'User Rating Count', 'Size', 
                                                                                         'Price', 'Icon URL']].head(10)
best.iloc[:, 0:-1]


# > ** 1. Cash, Inc. Fame & Fortune Game turns out to be best overall game with 5.0 rating and 374772 reviews **<br>
#   ** 2. There are also a lot of other Games with 5.0 rating and healthy number of reviews **

# In[ ]:


plt.figure(figsize=(5,5))
image = Image.open(urllib.request.urlopen(best.iloc[0, -1]))
plt.axis('off')
plt.imshow(image)
plt.show()


# > TO BE CONTINUED................................

# > * Any **Suggestions** would be appreciated <br>
# > * Please Consider To **Upvote** if this was helful
