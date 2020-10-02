#!/usr/bin/env python
# coding: utf-8

# ## Road to Viz Expert Series
# 
# ![app store](https://developer.apple.com/news/images/og/app-store-og.png)
# 
# I would like to record my practice to become an expert in data visualization. 
# 
# > And I finally become Kaggle Notebook Master!
# 
# ---
# 
# **Table of Contents**
# 
# - Import Default Library & Check Data
# - Simple Ideas Of Visualization
# - Check Missing Data (**`Missingno`**)
# - Name & Subtitle (**`Word Cloud`**)
# - Icon URL (**`Requests` & Crawling**)
# - Average User Rating (**`Bokeh`** : Countplot Compare)
# - Price & Rating (**`Seaborn`** : Regplot, Lmplot)
# - Primary Genre & Genres (**`NetworkX`**, Network Graph, seaborn heatmap)
# - Primary Genre & Genres (**`Squarify`**, Treemap)
# - Primary Genre & Genres (**`PyWaffle`**, Waffle Chart)
# - Original Release Date & Size (Time Series with bokeh)
# 
# 

# ## Import default Library & Check Data 
# <a id="#1"></a>
# 

# In[ ]:


# Data Processing
import numpy as np
import pandas as pd

# Basic Visualization tools
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 300
import seaborn as sns
sns.set_palette('husl')


# Bokeh (interactive visualization)
import bokeh
from bokeh.io import show, output_notebook
from bokeh.palettes import Spectral9
from bokeh.plotting import figure
output_notebook() # You can use output_file()

# Special Visualization
import wordcloud, missingno
from wordcloud import WordCloud # wordcloud
import missingno as msno # check missing value
import networkx as nx

# Check file list
import os
print(os.listdir('../input/17k-apple-app-store-strategy-games'))


# In[ ]:


# Version Check!!
print("matplotlib", matplotlib.__version__)
print("seaborn", sns.__version__)
print("bokeh", bokeh.__version__)
print("missingno", missingno.__version__)
print("wordcloud", wordcloud.__version__)
print("networkX", nx.__version__)


# In[ ]:


data = pd.read_csv('../input/17k-apple-app-store-strategy-games/appstore_games.csv')


# You can check the simple data information by using `describe` method.

# In[ ]:


data.describe() #numeric


# In[ ]:


data.describe(include='O') #categorical


# ## Simple Ideas of Visualization
# <a id="#2"></a>
# 
# 
# **Goal : Create a story with visualization**
# 
# ### Single Feature
# 
# Here's a simple idea about handling a single feature:
# 
# | Column Name                  | Type             | Simple Idea                     | Check|
# | ---------------------------- | ---------------- | ------------------------------- | ---- |
# | URL                          | Text(url)        | need to drop                    | O |
# | ID                           | Number(key_id)   | need to drop                    | O |
# | Name                         | Text (Title)     | Word Cloud                      | O |
# | Subtitle                     | Text (Sub_Title) | Word Cloud                      | O |
# | Icon URL                     | Text(url)        | Crawling Image                  | O |
# | Average User Rating          | Float(Score)     | Countplot                       | O |
# | User Rating Count            | Int(Counting)    | Outlier, Pie Chart              |   |
# | Price                        | Float(Price)     | Outlier, Pie Chart              | O |
# | In-app Purchases             | Float(Price)     | preprocessing, minmax graph     |   |
# | Description                  | Text             | need to drop                    |   |
# | Developer                    | Text             | WordCloud?                      |   |
# | Age Rating                   | Int(Ordinal)     | '+' remove                      |   |
# | Language                     | Text             | TreeMap, Countplot              |   |
# | Size                         | Int (Byte)       | Unit Converting, distplot       | O |
# | Primary Genre                | Text             | treemap                         | O |
# | Genres                       | Text             | treemp, network graph           | O |
# | Original Release Date        | Date             | Time Series                     | O |
# | Current Version Release Date | Date             | Time Series                     |   |
# 
# 
# ### Two or More Features
# 
# Let's make more stories in Data
# 
# - Price & User Rating Count : Could paid apps lead to user reactions?
# - Price & Age Rating : Price by age group
# - Genre & Size: Relationship between Genre and Size (add more features and make simple regression model)
# - Title & Subtitle & Genres : Tendency of titles by genre
# 

# In[ ]:


data = data.drop(['URL', 'ID'], axis=1)


# ## Check Missing Data (missingno)
# <a id="#3"></a>
# 
# `missingno` provides a small toolset of flexible and easy-to-use missing data visualizations and utilities that allows you to get a quick visual summary of the completeness (or lack thereof) of your dataset. 

# In[ ]:


msno.matrix(data)


# In[ ]:


print(data.columns)


# ## Name & Subtitle (Word Cloud)
# <a id="#4"></a>
# 
# 
# - Which words are most used
# 
# A **tag cloud** (**word cloud**, or weighted list in visual design) is a novelty visual representation of text data, typically used to depict keyword metadata (tags) on websites, or to visualize free form text.
# 
# Using WordCloud package, we can easily make workcloud image.
# 
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', "fig, ax = plt.subplots(1, 2, figsize=(16,32))\nwordcloud = WordCloud(background_color='white',width=800, height=800).generate(' '.join(data['Name']))\nwordcloud_sub = WordCloud(background_color='white',width=800, height=800).generate(' '.join(data['Subtitle'].dropna().astype(str)) )\nax[0].imshow(wordcloud)\nax[0].axis('off')\nax[0].set_title('Wordcloud(Name)')\nax[1].imshow(wordcloud_sub)\nax[1].axis('off')\nax[1].set_title('Wordcloud(Subtitle)')\nplt.show()")


# - **Name**
#     - Game, Free, War, Defense, Puzzle, Block, Chess
# - **Subtitle**
#     - Game, Classic, Battle, Puzzle, Best, Fun

# ## Icon URL (requests & Crawling)
# <a id="#5"></a>
# 
# This data is provided as a URL. In this case, you can get it by crawling. Please check your internet connection.
# 
# - single url crawling: 0.5 sec
# - total : 17000 * 0.5 = 8500 s = over 2 hour.. OMG 
# 
# just testing sample code (100 image)

# In[ ]:


get_ipython().run_cell_magic('time', '', "import matplotlib.pyplot as plt\nimport requests\nfrom PIL import Image\nfrom io import BytesIO\n\nfig, ax = plt.subplots(10,10, figsize=(12,12))\n\nfor i in range(100):\n    r = requests.get(data['Icon URL'][i])\n    im = Image.open(BytesIO(r.content))\n    ax[i//10][i%10].imshow(im)\n    ax[i//10][i%10].axis('off')\nplt.show()")


# ## Average User Rating (Bokeh : countplot compare)
# <a id="#6"></a>

# Bokeh don't have countplot. so we have to implement ad-hoc
# 
# 1. Use Pandas method : `value_counts()` 
# 2. Sort this values. Becauses `value_counts()` values are already sort by counting numbers. use `sort_index()`
# 3. x_range should be string list. so use `map` to convert index values

# In[ ]:


aur = data['Average User Rating'].value_counts().sort_index()
p = figure(x_range=list(map(str, aur.index.values)), 
           plot_height=250, title="Average User Rating", 
           toolbar_location=None, 
           tools="")

p.vbar(x=list(map(str, aur.index.values)), 
       top=aur.values, 
       width=0.9, 
       color=Spectral9)

p.xgrid.grid_line_color = None
p.y_range.start = 0
show(p)


# You can use this on **seaborn** and **matplotlib** like this

# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(16, 4))
sns.countplot(data['Average User Rating'],ax=ax[0]) # seaborn
ax[1].bar(aur.index, aur, width=0.4) # matplotlib
ax[1].set_title('Average User Rating')
plt.show()


# ## Price & Rating (seaborn : regplot, lmplot)
# <a id="#7"></a>

# In[ ]:


# price_column = list(map(str, data['Price'].value_counts().sort_index().index))
# rating_index = list(map(str, data['Average User Rating'].value_counts().sort_index().index))
# pr_table = pd.DataFrame(columns=price_column, index=rating_index)

# for price in price_column:
#     for rate in rating_index:
#         pr_table[price][rate] = len(data[(data['Price']==float(price) )& (data['Average User Rating'] == float(rate))])
        
# pr_table


# I want to look at the relationship between price and rating. Use regplot as a simple plot for this

# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(12, 7), dpi=72)
sns.regplot(data=data, x='Price', y='Average User Rating', ax=ax)
plt.show()


# As you can see, there seems to be little relationship between price and rating. Most of the ratings are 4 points.
# 
# How about mean and std value?

# In[ ]:


price_list = sorted(data['Price'].dropna().unique())
rating_stat = pd.DataFrame(columns=['mean', 'std', 'count'], index=price_list)
for price in price_list:
    tmp = data[data['Price']==price]['Average User Rating'].dropna()
    rating_stat['mean'][price] = tmp.mean()
    rating_stat['std'][price] = tmp.std()
    rating_stat['count'][price] = len(tmp)

rating_stat.T.head(len(price_list))


# There seems to be no tendency

# ## Primary Genre & Genres (Heatmap + networkx, Network Graph, Treemap)
# <a id="#8"></a>

# Not surprisingly, Games are the majority.
# 
# 
# 
# 

# In[ ]:


genre = data['Primary Genre'].value_counts()
p = figure(x_range=list(map(str, genre.index.values)), 
           plot_height=250, plot_width=1500, title="Primary Genre", 
           toolbar_location=None, 
           tools="")

p.vbar(x=list(map(str, genre.index.values)), 
       top=genre.values, 
       width=0.9, 
       color=Spectral9)

p.xgrid.grid_line_color = None
p.y_range.start = 0
show(p)


# How about sub-Genres?

# In[ ]:


data['Genres'].head()


# In[ ]:


data['GenreList'] = data['Genres'].apply(lambda s : s.replace('Games','').replace('&',' ').replace(',', ' ').split()) 
data['GenreList'].head()


# In[ ]:


gameTypes = []
for i in data['GenreList']: gameTypes += i
gameTypes = set(gameTypes)
print("There are {} types in the Game Dataset".format(len(set(gameTypes))))


# How do you know the correlation between them?
# 
# 1. heatmap : using `seaborn` heatmap
# 2. graph : using `networkx` and `plotly`

# ### heatmap (correlation)
# 
# A **heatmap** is a graphical representation of data where the individual values contained in a matrix are represented as colors.
# 
# Usually we use corr to calculate the correlation and draw it as a heatmap.
# 
# It is also effective for drawing contours from three-dimensional data.

# In[ ]:


from sklearn.preprocessing import MultiLabelBinarizer # Similar to One-Hot Encoding

test = data['GenreList']
mlb = MultiLabelBinarizer()
res = pd.DataFrame(mlb.fit_transform(test), columns=mlb.classes_, index=test.index)


# In[ ]:



corr = res.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(15, 14))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})


# ### Graph (corr)
# 
# NetworkX is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
# 
# **Reference**
# - [Visualising stocks correlations with Networkx](https://towardsdatascience.com/visualising-stocks-correlations-with-networkx-88f2ee25362e)

# In[ ]:


import networkx as nx

stocks = corr.index.values
cor_matrix = np.asmatrix(corr)
G = nx.from_numpy_matrix(cor_matrix)
G = nx.relabel_nodes(G,lambda x: stocks[x])
G.edges(data=True)

def create_corr_network(G, corr_direction, min_correlation):
    H = G.copy()
    for stock1, stock2, weight in G.edges(data=True):
        if corr_direction == "positive":
            if weight["weight"] <0 or weight["weight"] < min_correlation:
                H.remove_edge(stock1, stock2)
        else:
            if weight["weight"] >=0 or weight["weight"] > min_correlation:
                H.remove_edge(stock1, stock2)
                
    edges,weights = zip(*nx.get_edge_attributes(H,'weight').items())
    weights = tuple([(1+abs(x))**2 for x in weights])
    d = nx.degree(H)
    nodelist, node_sizes = zip(*d)
    positions=nx.circular_layout(H)
    
    plt.figure(figsize=(10,10), dpi=72)

    nx.draw_networkx_nodes(H,positions,node_color='#DA70D6',nodelist=nodelist,
                           node_size=tuple([x**2 for x in node_sizes]),alpha=0.8)
    
    nx.draw_networkx_labels(H, positions, font_size=8, 
                            font_family='sans-serif')
    
    if corr_direction == "positive": edge_colour = plt.cm.GnBu 
    else: edge_colour = plt.cm.PuRd
        
    nx.draw_networkx_edges(H, positions, edge_list=edges,style='solid',
                          width=weights, edge_color = weights, edge_cmap = edge_colour,
                          edge_vmin = min(weights), edge_vmax=max(weights))
    plt.axis('off')
    plt.show() 
    
create_corr_network(G, 'positive', 0.3)
create_corr_network(G, 'positive', -0.3)


# ### Treemap (counting based)
# 
# `squarify` can draw Treemap, but I prefer `plotly`'s treemap. ;)

# In[ ]:


import squarify
y = res.apply(sum).sort_values(ascending=False)[:20]
    
plt.rcParams['figure.figsize'] = (30, 10)
plt.style.use('fivethirtyeight')

squarify.plot(sizes = y.values, label = y.index)
plt.title('Top 30 Main Word', fontsize = 30)
plt.axis('off')
plt.show()


# ### Waffle Chart (counting base)
# 
# you can draw waffle chart by using `pywaffle`.

# In[ ]:


get_ipython().system('pip install pywaffle')
from pywaffle import Waffle


# In[ ]:


# type 2 : Auto-Size
fig = plt.figure(
    FigureClass=Waffle, 
    rows=13, 
    columns=21, 
    values=y,
    labels=["{}({})".format(a, b) for a, b in zip(y.index, y) ],
    legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.4), 'ncol': len(y)//3,  'framealpha': 0},
    font_size=20, 
    figsize=(12, 12),  
    icon_legend=True
)

plt.title('Waffle Chart : Genre distribution')

plt.show()


# ## Original Release Date & Size (time series with bokeh)
# <a id="#9"></a>
# 
# We will look at trends in app size over time.
# 
# 1. First, Convert `str` to `datetime` type
# 2. re-index by using `set_index`

# In[ ]:


data['Original Release Date'] = pd.to_datetime(data['Original Release Date'], format = '%d/%m/%Y')
date_size = pd.DataFrame({'size':data['Size']})
date_size = date_size.set_index(data['Original Release Date'])
date_size = date_size.sort_values(by=['Original Release Date'])
date_size.head()


# In[ ]:


date_size['size'] = date_size['size'].apply(lambda b : b//(2**10)) # B to KB


# Simple Plotting version.

# In[ ]:


fig = figure(x_axis_type='datetime',           
             plot_height=250, plot_width=750,
             title='Date vs App Size')
fig.line(y='size', x='Original Release Date', source=date_size)
show(fig)


# Let's look at the month for trends.
# 
# We can use `resample` method. 
# 
# The criteria for grouping depend on the parameters.
# 
# 'M' means 'end of month', 'Y' means 'Year'.

# In[ ]:


monthly_size = date_size.resample('M').mean()
tmp = date_size.resample('M')
monthly_size['min'] = tmp.min()
monthly_size['max'] = tmp.max()
monthly_size.head()


# In[ ]:


fig = figure(x_axis_type='datetime',           
             plot_height=250, plot_width=750,
             title='Date vs App Size (Monthly)')
fig.line(y='size', x='Original Release Date', source=monthly_size, line_width=2, line_color='Green')
show(fig)


# In[ ]:


yearly_size = date_size.resample('Y').mean()
monthly_size.head()
fig = figure(x_axis_type='datetime',           
             plot_height=250, plot_width=750,
             title='Date vs App Size (Monthly & Yearly)')
fig.line(y='size', x='Original Release Date', source=monthly_size, line_width=2, line_color='Green', alpha=0.5)
fig.line(y='size', x='Original Release Date', source=yearly_size, line_width=2, line_color='Orange', alpha=0.5)
show(fig)


# **If you like this notebook, please upvote :)**
