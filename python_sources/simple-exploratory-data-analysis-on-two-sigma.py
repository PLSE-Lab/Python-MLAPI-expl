#!/usr/bin/env python
# coding: utf-8

# ## How Does Stock Market Works?
# 
# A stock market, equity market or share market is the aggregation of buyers and sellers of stocks, which represent ownership claims on businesses; these may include securities listed on a public stock exchange, as well as stock that is only traded privately. To get this into easy understanding have a look at this infographics by  that gives breif overview
# While there has been enough evidence that news can change the market significantly. Taking one of the recent examples into light;  
# 1. [Facebook has lost 30% of its value since July](https://edition.cnn.com/2018/10/11/tech/facebook-stock-dip/index.html)
# 2. [Tesla Stock Is Crashing After SEC Fraud Lawsuit Against Elon Musk](http://fortune.com/2018/09/28/tesla-stock-elon-musk-sec-fraud/)
# Given these news  we can safely bail out on investing well before the market crashes. That's the very challenge we are solving in this competition. 
# 
# <br>
# 
# 
# <img src="http://www.visualcapitalist.com/wp-content/uploads/2017/02/stock-market-terms-share.png" width=780px>
#  <sub style="text-align: right;"> Source: Visual Capitalist </sub>

# In[ ]:


print('>> Importing libraries')
import os
import plotly
import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from kaggle.competitions import twosigmanews
from plotly.graph_objs import Scatter, Figure, Layout
print('>> Data Loading Completed!')

init_notebook_mode(connected=True)
cf.set_config_file(offline=True)
env = twosigmanews.make_env()


# In[ ]:


market_data, news_data = env.get_training_data()


# One of my favourite yet simple dataframe making tool to understand and have better glimpse of the frame. I call it Mr. Inspector as the function implies the same. Let's say hi.. to Mr. Inspector Frame

# In[ ]:


def mr_inspect(df):
    """Returns a inspection dataframe"""
    print ("Length of dataframe:", len(df))
    inspect_dataframe = pd.DataFrame({'dtype': df.dtypes, 'Unique values': df.nunique() ,
                 'Number of missing values': df.isnull().sum() ,
                  'Percentage missing': (df.isnull().sum() / len(df)) * 100
                 }).sort_values(by='Number of missing values', ascending = False)
    return inspect_dataframe


# 

# ## Exploring Market Data
#  Let's call on it on the martket data and have a quick analysis of the data given. All in one frame :)

# In[ ]:


mr_inspect(market_data)


# Personally, I am interested in latest trends than the past ones as these trends will have effect on the future data present in the test. The other trends are captured by the model however. We will keep this in mind when applying feature engineering on the data. Let's print out the head of the data frame and check out what's the range of dates given

# In[ ]:


market_data.head(10)
print ("The oldest date in dataset", market_data['time'].min())
print ("The latest date in dataset", market_data['time'].max())


# Now, let's dive the analysis of trends we see from 2007 - 2016. Diving the analysis into parts; 
# 1. The trend from 2007 - 2009
# 3. Trend from 2009 - 2016

#  ### Intitial Trend Analysis
# 

# In[ ]:


## Getting random 5 assets by Close Price for trend analysis
random_ten_assets = np.random.choice(market_data['assetName'].unique(), 5)
print(f"Using Top 10 Close Price Assets to Study The Trend:\n{random_ten_assets}")


# In[ ]:


data = []
for asset in random_ten_assets:
    asset_df = market_data[((market_data['assetName'] == asset) & (market_data['time'].dt.year <= 2009))]
    data.append(go.Bar(
        x = asset_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset_df['close'].values,
        name = asset
    ))
layout = go.Layout(dict(title = "Close Price of Random 5 assets overall for 2007-2009",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="h"))
iplot(dict(data=data, layout=layout), filename='basic-line')


# In[ ]:


data = []
for asset in random_ten_assets:
    asset_df = market_data[((market_data['assetName'] == asset) & (market_data['time'].dt.year >= 2009))]
    data.append(go.Bar(
        x = asset_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset_df['close'].values,
        name = asset
    ))
layout = go.Layout(dict(title = "Close Price of Random 5 assets overall for 2010-2016",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="h"))
iplot(dict(data=data, layout=layout), filename='basic-line')


# ### Let's check out the biggest drops 

# In[ ]:


market_data['diff'] = market_data['close'] - market_data['open']
groupedByAssetDiff = market_data.groupby('assetCode').agg({'diff': ['std', 'min']}).reset_index()


# In[ ]:


g = groupedByAssetDiff.sort_values(('diff', 'std'), ascending=False)[:10]
g['min_text'] = 'Maximum price drop: ' + (-1 * g['diff']['min']).astype(str)
trace = go.Scatter(
    x = g['assetCode'].values,
    y = g['diff']['std'].values,
    mode='markers',
    marker=dict(
        size = g['diff']['std'].values,
        color = g['diff']['std'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = g['min_text'].values
    #text = f"Maximum price drop: {g['price_diff']['min'].values}"
    #g['time'].dt.strftime(date_format='%Y-%m-%d').values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Top 10 Assests That Took A Shake!',
    hovermode= 'closest',
    yaxis=dict(
        title= 'diff',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
iplot(dict(data=data, layout=layout), filename='basic-line')


# If you have seen @artgor kernel. His analysis confirms that the data is but skeptical and might have noise. From his EDA,  the number of drop **9948** is confirmed from the above plot too. 
# 
# Let's look at the outliers in the target to see is they sound noisy too! 

# In[ ]:


data = []
market_data['month'] = market_data['time'].dt.month
for i in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
    price_df = market_data.groupby('month')['returnsOpenNextMktres10'].quantile(i).reset_index()
    data.append(go.Bar(
        x = price_df['month'].values,
        y = price_df['returnsOpenNextMktres10'].values,
        name = f'{i} quantile'
    ))
layout = go.Layout(dict(title = "Trends of grouby Month of returnsOpenNextMktres10 by 10 quartiles ",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="v"),)
iplot(dict(data=data, layout=layout), filename='basic-line')


# ## Choosing Apple Inc. As An Example Case

# In[ ]:


apple_data = market_data[market_data.assetCode == "AAPL.O"]
apple_data.head()


# In[ ]:


apple_data[['time','close']].set_index("time").iplot(title='APPLE CLOSE PRICE IN USD', 
                                                     theme='white',color='green', 
                                                     width=6, kind='bar')


# ### See that big drop in ** June 2014**. Let's Find Answers Though Web
# 
# ### Apple just got 'cheaper.' Will you buy?
# <img src ="https://i2.cdn.turner.com/money/dam/assets/140606111336-apple-stock-split-1024x576.png" width=400px; align=left;>
# 
# 
# I found searching on web from money.cnn;
# 
# > A share of Apple (AAPL) went from costing $645.57 (as of Friday's closing price) to about $92.44 -- give or take a few iCents. That's because the company did what's known as a stock split. It issued more shares to existing investors in order to bring down the price of the stock.
# 
# Here's the [Source Link](https://money.cnn.com/2014/06/06/investing/apple-stock-split/index.html)
# There are many hidden stories in just the numbers we see. I will leave out for you to figure out the rest of the stores. For now, lets move on with our EDA
# 

# ## Distributions of various Numerical Values

# In[ ]:


market_data.head()


# In[ ]:


import plotly.plotly as py
import plotly.figure_factory as ff
# Add histogram data
x1 = market_data.returnsClosePrevRaw1[:1000]
x2 = market_data.returnsOpenPrevRaw1[:1000] 

# Group data together
hist_data = [x1, x2]

group_labels = ['returnsClosePrevRaw1 1', 'returnsOpenPrevRaw1 2']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=.1)

# Plot!
iplot(fig, filename='Distplot with Multiple Datasets')


# In[ ]:


import plotly.plotly as py
import plotly.figure_factory as ff
# Add histogram data
x1 = market_data.returnsClosePrevRaw10[:1000]
x2 = market_data.returnsOpenPrevRaw10[:1000] 

# Group data together
hist_data = [x1, x2]
colors = ['#3A4750', '#F64E8B']

group_labels = ['returnsClosePrevRaw10', 'returnsOpenPrevRaw10']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=.1, curve_type='normal', colors=colors)

# Plot!
iplot(fig, filename='Distplot with Multiple Datasets')


# ## Sorting Target. Hunting Outliers!

# In[ ]:


hist_data = [market_data.returnsOpenNextMktres10.values[:5000]]
group_labels = ['distplot']
fig = ff.create_distplot(hist_data, group_labels)
iplot(fig, filename='Basic Distplot')


# **Plotly can't handle too many points in scatter plot. The kernel hangs itself. So, we will be swithching to Seaborn to plot if and any outliers in the data**

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(12,10))
plt.scatter(range(market_data.shape[0]), np.sort(market_data.returnsOpenNextMktres10.values), color='red')
plt.xlabel('index', fontsize=12)
plt.ylabel('returnsOpenNextMktres10', fontsize=12)
plt.show()


# ## Exploring News Data

# In[ ]:


#Statistics
from scipy.misc import imread
from scipy import sparse
import scipy.stats as ss

#More Viz Libs
import matplotlib.gridspec as gridspec 
from wordcloud import WordCloud ,STOPWORDS
from PIL import Image
import matplotlib_venn as venn

#For NLP
import spacy
import string
import re    #for regex
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer   


# Hope you still remember, our mr_inspect. He is ready to call and jump in right now... 

# In[ ]:


mr_inspect(news_data)


# Let's print out the head frame now... 

# In[ ]:


news_data.head()


# The feature - `urgency(int8) - differentiates story types (1: alert, 3: article)` from the description interests me. Let's explore this further by categorizing them
# 

# In[ ]:


alerts_df = news_data[news_data.urgency == 1]
article_df = news_data[news_data.urgency == 3]


# In[ ]:


eng_stopwords  = list(STOPWORDS)


# In[ ]:


# alertnews
#wordcloud for alerts comments
text=alerts_df.headline.values[:1000]
wc= WordCloud(background_color="black",max_words=2000,stopwords=eng_stopwords)
wc.generate(" ".join(text))
plt.figure(figsize=(20,10))
plt.axis("off")
plt.title("Words frequented in Alert Headlines", fontsize=30)
plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)
plt.show()


# In[ ]:


#wordcloud for articles comments
text=article_df.headline.values[:1000]
wc= WordCloud(background_color="black", max_words=2000, stopwords=eng_stopwords)
wc.generate(" ".join(text))
plt.figure(figsize=(20,10))
plt.axis("off")
plt.title("Words frequented in Article Headlines", fontsize=30)
plt.imshow(wc.recolor(colormap= 'inferno' , random_state=17), alpha=0.98)
plt.show()


# ### This kernel will be further updated on Numerical And Surface EDA on News Dataset. There is kernel coming soon on extensive feature engineering with News Data. The advanced EDA will be continued there as the kernel hangs itself after any analysis on text at this point. 
# 
# Thanks! Let me know what you think about the current version
# 
# *WIP
# shaz13*
