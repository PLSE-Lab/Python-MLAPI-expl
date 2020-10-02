#!/usr/bin/env python
# coding: utf-8

# ### Lets import some basic libraries and read the data

# In[ ]:


import pandas as pd
data=pd.read_csv("../input/top50spotify2019/top50.csv", encoding='cp1252')
data.head(10) #looking at the top 10 variables


# In[ ]:


data.info()


# ### Lets rename some columns and have some fun with this dataset

# In[ ]:


df = data # Making a copy of my data
df1 = df.rename(columns={"Artist.Name": "Artist", "Track.Name": "Track"})
df1.info()


# In[ ]:


# Performing a quick visualization of the most popular Artist Name using WordCloud
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
string=str(df1.Artist)
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='white',
                      width=1000,
                      height=1000).generate(string)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


# Performing a quick visualization of the most popular Genre using WordCloud
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
string=str(df1.Genre)
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='white',
                      width=1000,
                      height=1000).generate(string)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


# Performing a quick visualization of the most popular Track Name using WordCloud
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
string=str(df1.Track)
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='white',
                      width=1000,
                      height=1000).generate(string)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


#detailed visualization of relationship between genre and popularity using SwarmPlot
#genre with most popular song
#genre/genres with most/least number of popular songs
#genre/genres with most/least variance in popularity of songs
import seaborn as sns
plt.figure(figsize=(10,5))
swarmplot=sns.swarmplot(x="Genre",y="Popularity",data=df1)
swarmplot.set_xticklabels(swarmplot.get_xticklabels(),rotation=90)
swarmplot.set_title("Relationship between Genre & Popularity")


# In[ ]:


#visualizing relationship between danceability and popularity using RegPlot
regplot=sns.regplot(x="Danceability",y="Popularity",data=df1)
regplot.set_title("Relationship between danceability and popularity")


# ### Im noticing a trend in the data as there has to be a better was to figure out different patterns

# In[ ]:


#detailed visualization of relationship between genre and popularity using SwarmPlot
#genre with most popular song
#genre/genres with most/least number of popular songs
#genre/genres with most/least variance in popularity of songs
import seaborn as sns
plt.figure(figsize=(10,5))
swarmplot=sns.swarmplot(x="Artist",y="Popularity",data=df1)
swarmplot.set_xticklabels(swarmplot.get_xticklabels(),rotation=90)
swarmplot.set_title("Relationship between Artist & Popularity")


# In[ ]:


#visualizing relationship between danceability and Length using RegPlot
regplot=sns.regplot(x="Danceability",y="Length.",data=df1)
regplot.set_title("Relationship between Danceability and Length of a song")


# # Lets run further analysis for fun

# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')

df1.plot()  # plots all columns against index
df1.plot(kind='scatter',x='Danceability',y='Length.') # scatter plot
df1.plot(kind='density')  # estimate density function
# df.plot(kind='hist')  # histogram


# # Lets see a correlation matrix

# In[ ]:


df1.corr()


# In[ ]:


from pandas.plotting import scatter_matrix

scatter_matrix(df1, alpha=0.2, figsize=(15, 15))
plt.show()


# In[ ]:




