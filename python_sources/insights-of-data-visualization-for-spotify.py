#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[ ]:


songs = pd.read_csv('/kaggle/input/top50spotify2019/top50.csv',encoding='ISO-8859-1')
songs.head(5)


# In[ ]:


new_songs = songs.drop(['Unnamed: 0'],axis=1)


# In[ ]:


new_songs.describe()


# In[ ]:


new_songs.info()


# In[ ]:


## Check on Target Column--Popularity
plt.figure(figsize=(8,6))
plt.scatter(range(new_songs.shape[0]),np.sort(new_songs['Popularity'].values))
plt.xlabel('Row Index')
plt.ylabel('Popularity Count')
plt.show()


# In[ ]:


#Distribtution of every feature with target feature--here popularity#


# In[ ]:


## all Feature check with Popularity 
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18,10))
axes = axes.flatten()
colors="bgrcmykw"
import random
num_cols = list(new_songs.select_dtypes([np.number]).columns[:-1])
#selecting all the numeric columns except Popularity
plt.tight_layout(pad=2)
for i, j in enumerate(num_cols):
    axes[i].scatter(x=new_songs[j], y=new_songs['Popularity'], color= random.choice(colors), edgecolor='black')
    axes[i].set_xlabel(j)
    axes[i].set_ylabel('Popularity')


# In[ ]:


plot gives us indication that songs with most energy and most danceability are most popular


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x='Artist.Name',data=new_songs,color='c')
plt.xlabel('Artist.Name in given data',fontsize=12)
plt.ylabel('count of song sung',fontsize=12)
plt.xticks(rotation='vertical')
plt.title('Artist.Name and no.of song count',fontsize=12)
plt.show()


# In[ ]:


Ed Sheeran has sung maximum number of songs


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x='Genre',data=new_songs,color='b')
plt.xlabel('Songs Genre',fontsize=12)
plt.ylabel('count of Genre',fontsize=12)
plt.xticks(rotation='vertical')#Summary statistic of all Genre
plt.title('Songs Genre and no.of song count',fontsize=12)
plt.show()


# In[ ]:


seems that most of the songs are from 'dance pop' genre


# In[ ]:


Summary statistic of all Genre


# In[ ]:


# Groupby by Genre
genre = new_songs.groupby("Genre")
genre.describe().head()
genre.mean().sort_values(by="Popularity",ascending=False).head()
plt.figure(figsize=(15,10))
genre.size().sort_values(ascending=False).plot.bar()
plt.xticks(rotation=50)
plt.xlabel("songs genre")
plt.ylabel("Polulaity")
plt.show()


# In[ ]:


##Let's now take a look at the plot of all genre by its highest rated genre,
##using the same plotting technique as above:
plt.figure(figsize=(15,10))
genre.max().sort_values(by="Popularity",ascending=False)["Popularity"].plot.bar()
plt.xticks(rotation=50)
plt.xlabel("songs genre")
plt.ylabel("Polulaity")
plt.show()


# In[ ]:


'electro pop'is most populare genre amongst in the list of top 50


# In[ ]:


plt.figure(figsize=(10,8))
sns.jointplot(x=songs['Danceability'], y=songs['Popularity'].values, size=10, kind="reg",color='g')
plt.ylabel('Popularity', fontsize=12)
plt.xlabel("Danceability", fontsize=12)
plt.title("Danceability Vs Popularity", fontsize=15)
plt.show()


# In[ ]:


sns.pairplot(songs, x_vars=['Beats.Per.Minute','Energy'], y_vars=(['Popularity']),height=5, kind='reg')


# In[ ]:


sns.pairplot(songs, x_vars=['Beats.Per.Minute','Speechiness.'], y_vars=(['Popularity']),height=5, kind='reg')


# In[ ]:


pairplot shows 'beats per sec' and 'speechiness' showing some lenearity with popularity


# In[ ]:


#Finding out the skew for each attribute
skew=new_songs.skew()
print(skew)


# In[ ]:


Skewness indicates whether the data is concentrated on one side


# In[ ]:


# Removing the skew by using the boxcox transformations
from scipy import stats
transform=np.asarray(songs['Liveness'].values)
songs_transform = stats.boxcox(transform)[0]
# Plotting a histogram to show the difference 
plt.hist(songs['Liveness'],bins=10) #original data
plt.show()
plt.hist(songs_transform,bins=10) #corrected skew data
plt.show()


# In[ ]:


sns.distplot(new_songs['Popularity'],kde=True,color='blue')
plt.show()


# In[ ]:


transform=np.asarray(new_songs['Popularity'].values)
songs_transform = stats.boxcox(transform)[0] 
sns.distplot(songs_transform,kde=False,color='red')
plt.show()


# In[ ]:


degree of skewness before and after in 'popularity' and 'Liveliness' attributes


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))        
sns.heatmap(new_songs.corr(),annot=True,linewidths=.5, ax=ax)


# In[ ]:


def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)


# In[ ]:


labels1 = songs['Artist.Name'].value_counts().index
sizes1 = songs['Artist.Name'].value_counts().values
colors = ['purple', 'yellow','lightcoral', 'lightskyblue','cyan', 'green', 'blue','yellow','orange','pink']
plt.figure(figsize = (13,10))
plt.pie(sizes1, labels=labels1, autopct=lambda pct: func(pct, sizes1),colors=colors)
plt.axis('equal')
plt.show()


# In[ ]:


##distributions and proportions of each genre on the top 50 list
# calculate the number of tracks by genre
Genre_counts = new_songs["Genre"].value_counts()
Genre_counts_index = Genre_counts.index
Genre_counts, Genre_counts_index = zip(*sorted(zip(Genre_counts, Genre_counts_index)))


# In[ ]:


# treemap for visualizing proportions
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly
import plotly.graph_objects as go
plotly.offline.init_notebook_mode(connected=True)
fig = go.Figure(
    go.Treemap(
        labels = ["Number of Tracks by Genre of the Spotify Top 50 Music List"] + list(Genre_counts_index),
        parents = [""] + ["Number of Tracks by Genre of the Spotify Top 50 Music List"] * len(Genre_counts_index),
        values = [0] + list(Genre_counts),
        textposition='middle center', # center the text
        textinfo = "label+percent parent", # show label and its percentage among the whole treemap
        textfont=dict(
            size=12 # adjust small text to larger text
        )
    )
)
#plotly.offline.iplot(fig, filename= treemap + ".html")
fig.show()
#import plotly.plotly as py
# Save the figure as a png image:
#py.image.save_as(fig, 'my_plot.png')


# In[ ]:


from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


# Start with one review:
##visualization of most popular genre group using WordCloud
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
string=str(songs.Genre)
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='cyan',
                      width=4000,
                      height=2000).generate(string)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()


# # These are the ways of visualization to Discover Insights in spotify dataset.
# # 

# 
