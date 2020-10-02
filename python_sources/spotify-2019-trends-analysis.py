#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
data=pd.read_csv("../input/top50spotify2019/top50.csv", encoding='cp1252')
data.head()


# In[ ]:


#quick visualization of most popular genre group/sub group using WordCloud
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
string=str(data.Genre)
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
swarmplot=sns.swarmplot(x="Genre",y="Popularity",data=data)
swarmplot.set_xticklabels(swarmplot.get_xticklabels(),rotation=90)
swarmplot.set_title("Relationship between Genre & Popularity")


# In[ ]:


#visualizing relationship between danceability and popularity using RegPlot
regplot=sns.regplot(x="Danceability",y="Popularity",data=data)
regplot.set_title("relationship between danceability and popularity")

