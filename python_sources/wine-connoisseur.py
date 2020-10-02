#!/usr/bin/env python
# coding: utf-8

# <center><h3>Let's Talk About Wine</h3></center>
# <img src="https://i.ibb.co/X8vkKyx/jean-luc-benazet-761840-unsplash.jpg" width="50%">
# <br/>
# <center>
# This database contains 10 columns and 130k rows of wine reviews. <br/>
# I want to test three hypotheses: 
# </center> 
# <ul>
#     <li><b>Hypothesis One: </b> High price does not guarantee high wine points.</li>
#     <li><b>Hypothesis Two: </b> Sentiment scores of wine description would have a high correlation to the wine points. 
#     </li>
# <li><b>Hypothesis Three: </b> Testers have biases based on either specific wineries or regions. 
#     </li>
# </ul>

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os


# In[ ]:


df = pd.read_csv("../input/winemag-data-130k-v2.csv",sep = ",")
df.head(5)


# We will determine the columns that are relevant and then drop any rows that have missing values.

# In[ ]:


columns = ['country', 'description', 'points', 'price', 'province', 'taster_name', 'title', 'variety', 'winery']
wine_reviews = df[columns].dropna(axis=0, how='any')
wine_reviews.info()


# No empty values, our data is ready!

# In[ ]:


wine_reviews.head(5)


# <h3><b>Hypothesis One: </b> High price does not guarantee high wine points.</h3>

# In[ ]:


fig = plt.figure(figsize=(16,4))
sns.scatterplot(x='points', y='price', hue='price', data=wine_reviews)


# In[ ]:


wine_reviews.corr()


# Based on the scatterplot and correlation table, we can see that there is <b>a moderate correlation between price and points. </b> However we will look into the data and observe behaviorial patterns based on two factors: (1) Highest/Lowest price, (2) Highest/Lowest points.  

# In[ ]:


# Highest priced wine
wine_reviews.loc[wine_reviews['price']==wine_reviews.price.max()]


# In[ ]:


# Lowest priced wine
min_price = wine_reviews.loc[wine_reviews['price']==wine_reviews.price.min()]
print(min_price.mean())
min_price


# The highest priced wine (3300) had 88 points, whereas the lowest priced wines (4) on average received 84 points. <br/>
# <b>In fact, that is only 4 points lower than the most expensive wine with a $3296 price difference. </b>

# In[ ]:


print("Max Points:", wine_reviews.points.max())
print("There are", len(wine_reviews.loc[wine_reviews['points']==100]), "wines with the highest point of 100.")
print("Min Points:", wine_reviews.points.min())
print("There are", len(wine_reviews.loc[wine_reviews['points']==80]), "wines with the lowest point of 80.")


# In[ ]:


max = wine_reviews.loc[wine_reviews['points']==100]
min = wine_reviews.loc[wine_reviews['points']==80]

fig = plt.figure(figsize=(16,4))
fig.suptitle("Count of Price Ranges For Wines with Lowest Points (80)")
sns.countplot(min['price'], palette='spring')
fig2 = plt.figure(figsize=(16,4))
fig2.suptitle("Count of Price Ranges For Wines with Highest Points (100)")
sns.countplot(max['price'], palette='summer')


# For wines with lowest points (80), we can see a slightly skewed graph to the right. Wines with the highest points (100) was randomly distributed, but <b>the price difference is notable (1420). </b>
# In conclusion, I was wrong with my initial hypothesis: <b> Price and points do have a positive correlation in this dataset.</b> However there were outliers that demonstrated high price does not always guarantee high points. 

# <h3><b>Hypothesis Two: </b> Sentiment scores of wine description would have a high correlation to the wine points. </h3>

# The 'description' column contains wine reviews. We will draw sentiment scores from that column and see if there is a high correlation to 'wine points'.

# In[ ]:


# Review sample
wine_reviews['description'].values[0]


# In[ ]:


# Lower-case wine reviews
wine_reviews['description'] = wine_reviews['description'].apply(lambda x: " ".join(x.lower() for x in x.split()))
wine_reviews['description'].head(5)


# In[ ]:


# Remove punctuations from wine reviews
wine_reviews['description'] = wine_reviews['description'].str.replace('[^\w\s]','')
wine_reviews['description'].head()


# In[ ]:


# Remove stop words from wine reviews
from nltk.corpus import stopwords
stop = stopwords.words('english')
wine_reviews['description'] = wine_reviews['description'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
wine_reviews['description'].head()


# In[ ]:


# Remove common words from wine reviews
freq = pd.Series(' '.join(wine_reviews['description']).split()).value_counts()[:10]
freq


# In[ ]:


# Remove common words 
freq = list(freq.index)
wine_reviews['description'] = wine_reviews['description'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
wine_reviews['description'].head()


# In[ ]:


# Spell-check wine reviews
from textblob import TextBlob
wine_reviews['description'][:5].apply(lambda x: str(TextBlob(x).correct()))


# In[ ]:


# Lemmatize wine reviews
from textblob import Word
wine_reviews['description'] = wine_reviews['description'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
wine_reviews['description'].head()


# In[ ]:


# Sentiment Analysis 
def detect_polarity(text):
    return TextBlob(text).sentiment.polarity

wine_reviews['polarity'] = wine_reviews['description'].apply(detect_polarity)
wine_reviews.head(3)


# In[ ]:


max_pol = wine_reviews.polarity.max()
min_pol = wine_reviews.polarity.min()
print("Maximum polarity score is", max_pol, "while the minimum is", min_pol)


# Now that we have the polarity calculated, we will draw charts to see the big picture: 

# In[ ]:


fig = plt.figure(figsize=(16,4))

sns.distplot(wine_reviews['polarity'], color='y')


# In[ ]:


fig = plt.figure(figsize=(16,6))
color = sns.cubehelix_palette(21, start=.5, rot=-.85)
sns.boxenplot(x='points', y='polarity', data=wine_reviews, palette=color)


# In[ ]:


wine_reviews['polarity'].corr(wine_reviews['points'])


# The correlation is weak as it is only 0.168. <b>Therefore, my initial hypothesis that wine description sentiments scores would have a high correlation to points was proven wrong. </b>However, we can see on the chart above that as points incremented, the mean polarity score consistently grew.  

# <h3><b>Hypothesis Three: </b> Testers have biases.</h3>

# In[ ]:


len(wine_reviews['taster_name'].unique())


# There are 19 wine reviewers/tasters. We will review the data through their eyes. 

# In[ ]:


wine_reviews.sort_values(by='taster_name', inplace=True)
wine_reviews.groupby('taster_name').agg(['count'])


# We can see a huge variation in the number of reviews each reviewer contributed, from 6 to 20,172, so we will look into the two extremes - the lowest (Christina Pickard) and highest (Roger Voss) to see if any bias exists. <br/>
# 

# In[ ]:


wine_reviews.loc[wine_reviews.taster_name == 'Christina Pickard']


# Christina Pickard's data is too small (6) for any real bias picking, but it's interesting to see that <b>the only wine she gave above 90 points was Australian </b>while the rest she reviewed were Californian. 
# Roger Voss was the most active reviewer, with 20,172 reviews in total, so we will try to analyze his patterns if there are any.

# In[ ]:


# Roger Voss country data 
roger_voss = wine_reviews.loc[wine_reviews['taster_name']=='Roger Voss']
df = roger_voss.groupby(['country'],as_index=False).agg(['count']).reset_index()
df.columns = df.columns.droplevel(1)
df = df[['country', 'description']]
df.rename(columns={'description':'count'}, inplace=True)
df.sort_values('count', ascending=False, inplace=True)
df = df.reset_index(drop=True)
list_country = df['country'].tolist()
list_count = df['count'].tolist()


# In[ ]:


import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

data = [go.Choropleth(
           locations = list_country,
           locationmode = 'country names',
           z = list_count,
           colorscale = 'Greens',
           autocolorscale = False,
           reversescale = True,
           marker = dict(line = dict(color = 'rgb(255,255,255)', width=1)),
           )]

layout = dict(title = 'Roger Voss Wine Reviews by Countries',
             geo = dict(scope = 'world'))
choromap = go.Figure(layout=layout, data=data)
iplot(choromap)


# In[ ]:


x = 14395/20172
y = 20153/20172
print("French wine reviews:" + " {:.2%}".format(x))  
print("European wine reviews:" + " {:.2%}".format(y))


# Roger Voss wrote 20,172 wine reviews, and <b>99.91% were European. </b> Therefore we can assume that he is experienced with European wines, especially French (14,395 reviews in total). In comparison, he only wrote 2 reviews for American wines. 

# In[ ]:


fig = plt.figure(figsize=(16,3))
color = sns.color_palette('summer', 20)
sns.countplot(x='points', data=roger_voss, palette=color)


# It's not surprising to see a bell-curve forming for 'points' count, but what's unusual is the sudden drop for 89. 
# <img src="https://i.ibb.co/w7FLZk1/drop.png"><br />
# It is the only score range that doesn't follow the natural flow. <b>This makes me question: 'What causes the distinction?'</b> 

# In[ ]:


data = roger_voss.loc[(roger_voss.points >= 88) & (roger_voss.points <= 90)].sort_values(by='points')
color = sns.color_palette('spring', 3)
fig = plt.figure(figsize=(16,6))
sns.countplot(x='country', hue = 'points', data=data, palette=color)


# Of course, there are a lot of factors affecting the trend, but it's interesting to see <b> an identical pattern with 89 point review counts dropping drastically for both French and Portugese wine review counts. </b> 

# In[ ]:


# Getting winery review counts for 88 point wines 
wine_88 = data.loc[data.points==88]
wine_88 = wine_88.groupby(by='winery',as_index=False).count().sort_values(by='country',ascending=False).head(10).reset_index(drop=True)
list_winery = wine_88['winery'].tolist()
count_winery = wine_88['country'].tolist()
d = {'Winery': list_winery, 'Count': count_winery}
df = pd.DataFrame(d)
df['Points'] = '88'


# In[ ]:


# Getting winery review counts for 89 point wines 
wine_89 = data.loc[data.points==89]
wine_89 = wine_89.groupby(by='winery',as_index=False).count().sort_values(by='country',ascending=False).head(10).reset_index(drop=True)
wine_89
list_winery_two = wine_89['winery'].tolist()
count_winery_two = wine_89['country'].tolist()
d_two = {'Winery': list_winery_two, 'Count': count_winery_two}
d_two = pd.DataFrame(d_two)
d_two['Points'] = '89'
df = df.append(d_two)


# In[ ]:


# Getting winery review counts for 90 point wines 
wine_90 = data.loc[data.points==90]
wine_90.groupby(by='winery',as_index=False).count().sort_values(by='country',ascending=False).head(10).reset_index(drop=True)
list_winery_three = wine_89['winery'].tolist()
count_winery_three = wine_89['country'].tolist()
d_three = {'Winery': list_winery_three, 'Count': count_winery_three}
d_three = pd.DataFrame(d_three)
d_three['Points'] = '90'
df = df.append(d_three)


# In[ ]:


df.loc[df['Winery']=='DFJ Vinhos']


# In[ ]:


fig = plt.figure(figsize=(16,8))
sns.lineplot(x='Points', y='Count', hue='Winery', data=df)
plt.title('Reviews by Roger Voss')


# Based on the chart above, wineries such as <b> DFJ Vinhos, Wines &amp; Winemakers and Manuel Olivier </b> had the most reviews with 88 point wines, but the numbers drastically dropped for 89 and 90 points. <b>For DFJ Vinhos, the number dropped from 27 to 11, with 59% decrease. </b>
