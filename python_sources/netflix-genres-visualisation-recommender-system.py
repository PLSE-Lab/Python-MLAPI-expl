#!/usr/bin/env python
# coding: utf-8

# # Netflix Data Visualization and Recommender System
# - Data Visualization using Squarify and simple Matplotlib
# - Recommender System using Rake, CountVectorizer, CosinusSimiliarity

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly
import seaborn as sb
import squarify
import plotly.offline as py
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer 
import matplotlib.style as style 
get_ipython().system('pip install rake-nltk')
from rake_nltk import Rake


# In[ ]:


df = pd.read_csv("../input/netflix-shows/netflix_titles.csv")
df.head(5)


# # 1. Data Visualization

# In[ ]:


plt.figure(figsize=(6,6))
plt.pie([df.groupby('type').show_id.count()['Movie'],df.groupby('type').show_id.count()['TV Show']],
        labels = ['Movie', 'TV Shows'], colors=['mediumseagreen', 'khaki'],
       startangle = 0, shadow=True, counterclock=False,
       textprops = {'size':13}, autopct = '%1.1f%%')
centercircle = plt.Circle((0,0), 0.7, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centercircle)


plt.title('Type of Content', fontsize=15, fontweight='bold')
plt.show()


# ## A. Proportion of Genre in Movies and TV Show

# Getting the unique values for Movie Genre 

# In[ ]:


genremovies = []
for i in df[df['type']=='Movie'].listed_in.apply(lambda x: x.split(',')):
    genremovies = genremovies + i

allgenremovies = list(map(lambda x: x[1:] if x[0]==' ' else x, genremovies))
genremovies = list(set(allgenremovies))
genremovies.sort()


# Getting the unique values for TV Shows Genre 

# In[ ]:


genretv = []
for i in df[df['type']=='TV Show'].listed_in.apply(lambda x: x.split(',')):
    genretv = genretv + i

allgenretv = list(map(lambda x: x[1:] if x[0]==' ' else x, genretv))
genretv = list(set(allgenretv))
genretv.sort()


# Counting the number for each genre

# In[ ]:


ngenmov= []
for i in genremovies:
    ngenmov.append(allgenremovies.count(i))
    
ngentv= []
for i in genretv:
    ngentv.append(allgenretv.count(i))


# Making the label

# In[ ]:


labelmov= []
for i in range(len(ngenmov)):
    labelmov.append(genremovies[i] + ' ' +str(round(ngenmov[i]/len(allgenremovies)*100, 2)) + '%')
    
labeltv= []
for i in range(len(ngentv)):
    labeltv.append(genretv[i] + ' ' + str(round(ngentv[i]/len(allgenretv)*100, 2)) + '%')


# Squarify Visualization

# In[ ]:


plt.figure(figsize=(16,16))

plt.suptitle('Genres in Netflix', fontsize=20, fontweight='bold')
plt.subplot(2,1,1)
plt.title('Genre of Movies in Netflix', fontsize=14, fontweight='bold')
squarify.plot(sizes=ngenmov, label=labelmov, alpha=.6, edgecolor="black",
              linewidth=2, text_kwargs={'fontsize':12})

plt.subplot(2,1,2)
plt.title('Genre of TV Show in Netflix', fontsize=14, fontweight='bold')
squarify.plot(sizes=ngentv, label=labeltv, alpha=.6, edgecolor="black",
              linewidth=2, text_kwargs={'fontsize':12})

plt.show()


# ## B. Growth of The Genres

# To see the trend over the year, we need to see add a column of year added

# In[ ]:


df =df.dropna(subset=['date_added'])
df.head(3)


# In[ ]:


df['date_added']=df.date_added.apply(lambda x: datetime.strptime(x, ' %B %d, %Y') if x[0]==' ' else datetime.strptime(x, '%B %d, %Y'))
df['date_added']=df.date_added.apply(lambda x: datetime.strftime(x, '%B %d, %Y'))
df['date_added']=df.date_added.apply(lambda x: datetime.strptime(x, '%B %d, %Y'))


# In[ ]:


df['year_added']=df.date_added.apply(lambda x: x.year)


# In[ ]:


df.head(3)


# <b>Number of Movies Released per Genre per Year </b>

# In[ ]:


dfym = pd.DataFrame()
dfym['genre'] = genremovies
yearmovie = list(df.year_added.unique())
yearmovie.sort()

for i in yearmovie:
    dfym[f'{i}'] = dfym['genre'].apply(lambda x: list(df.groupby(['year_added', 'type']).agg({'listed_in': ','.join}).loc[i].loc['Movie'])[0].replace(', ', ',').split(',').count(x))
    
dfym


# <b>Number of TV Shows Released per Genre per Year </b>

# In[ ]:


dfytv = pd.DataFrame()
dfytv['genre'] = genretv

yeartv = list(df[df['type']=='TV Show']['year_added'])

for i in yearmovie:
    if i in yeartv: 
        dfytv[f'{i}'] = dfytv['genre'].apply(lambda x: list(df.groupby(['year_added', 'type']).agg({'listed_in': ','.join}).loc[i].loc['TV Show'])[0].replace(', ', ',').split(',').count(x))
    else:
        dfytv[f'{i}'] = [0] * len(genretv)
dfytv


# <b>Accumulate</b> the number of <b>movies</b> per genre per year

# In[ ]:


dfymA = dfym.copy()


# In[ ]:


for i in range(2,len(dfymA.columns)):
    dfymA[f'{dfymA.columns[i]}'] = dfym[f'{dfymA.columns[i]}'] + dfymA[f'{dfymA.columns[i-1]}']
dfymA


# <b>Accumulate</b> the number of <b>TV Shows</b> per genre per year
# 

# In[ ]:


dfytvA = dfytv.copy()


# In[ ]:


for i in range(2,len(dfytvA.columns)):
    dfytvA[f'{dfytvA.columns[i]}'] = dfytv[f'{dfytvA.columns[i]}'] + dfytvA[f'{dfytvA.columns[i-1]}']
dfytvA


# <b>Plotting </b>

# In[ ]:


plt.figure(figsize=(16,16))
plt.style.use('ggplot')
plt.suptitle('Rise of Genres in Netflix', family='sans-serif', size=20)


plt.subplot(2,1,1)
plt.title('Movie')
for i in range(len(genremovies)): 
    plt.scatter(range(2008,2021),dfymA.iloc[i, 1:], marker= 'D', s=30, color='yellow', edgecolor='black', zorder = 3)
    plt.plot(range(2008,2021), dfymA.iloc[i, 1:])
plt.xticks(range(2008,2021))
plt.grid(linestyle='--')
plt.legend(dfymA['genre'])



plt.subplot(2,1,2)
plt.title('TV Shows')
for i in range(len(genretv)): 
    plt.plot(range(2008,2021), dfytvA.iloc[i, 1:])
    plt.scatter(range(2008,2021),dfytvA.iloc[i, 1:], marker= 'D', s=30, color='yellow', edgecolor='black', zorder = 3)
plt.xticks(range(2008,2021))
plt.grid(linestyle='--')
plt.legend(dfytvA['genre'])



plt.show()


# Based on the graphic above we can see that Netflix started adding TV Shows and Movies in larger scale in 2016. Movie genre that Netflix added the most is Classic Movies, while on the TV Show Netflix aggresively adds the International TV Shows genre. From the line chart above we can also see that there hasn't been any big changes in the increase of movies/tv shows added per genre. Netflix adds more movies and tv shows in each genres rather steadily. 

# In[ ]:





# # 2. Recommender System

# ### Based on Similiarity

# We're going to build a simple recommender system based on the similiarity of a movie/tv show genre, description, cast, and directors. 

# In[ ]:


dfr = df.copy()
dfr.head(2)


# <b>Data Cleaning for Director, Cast, Genre</b><br> 
# eliminating the space between first name and last name, or 2-or-more-words-genre so that the extractor wouldn't split them into two different keywords.

# In[ ]:


dfr['cast'] = dfr['cast'].apply(lambda x: str(x).replace(' ', ''))
dfr['cast'] = dfr['cast'].apply(lambda x: str(x).replace(',',' '))

dfr['director'] = dfr['director'].apply(lambda x: str(x).replace(' ', ''))
dfr['director'] = dfr['director'].apply(lambda x: str(x).replace(',',' '))

dfr['genre'] = dfr['listed_in'].apply(lambda x: str(x).replace(' ', ''))
dfr['genre'] = dfr['genre'].apply(lambda x: str(x).replace(',',' '))


# In[ ]:


dfr.head(2)


# Using <b>Rake</b> to extract the keywords from the <b>description</b> since the description is too long

# In[ ]:


keyw = []
for index, row in dfr.iterrows():
    desc = row['description']
    r = Rake()
    r.extract_keywords_from_text(desc)
    keywordsdict = r.get_word_degrees()
    keyw.append(' '.join(list(keywordsdict.keys())))
    
dfr['keywords']= keyw


# In[ ]:


dfr.drop(['duration', 'description', 'listed_in'], axis =1, inplace=True)


# <b>Bagging The Keywords </b>

# In[ ]:


dfr['keybag'] = dfr['cast'] + ' ' + dfr['director'] + ' ' + dfr['genre'] + ' ' + dfr['keywords']


# In[ ]:


dfr2 = dfr.copy()
dfr2 = dfr2[['show_id','title', 'keybag', 'type',  'rating']]
dfr2['title2'] = dfr2['title'].apply(lambda x: x.lower())
dfr2['type'] = dfr2['type'].apply(lambda x: x.lower())


# In[ ]:


dfr2.head(3)


# <b>CountVectorizer</b>

# In[ ]:


extractor = CountVectorizer()
extracted = extractor.fit_transform(dfr2['keybag'])


# <b>Cosinus Similiarity</b>

# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity 

score = cosine_similarity(extracted)


# <b>Recommendation Function </b>

# We're gonna make a simple function named <b>recnflix</b> which will recommend you several tv show/movies based on the similiarity to your favorite tv show/movies. In this function you can choose how many tv shows/movies you want to get recommended to, and whether you want the recommendation to include TV Show/Movies. If you call the function it will return a dataframe that shows the recommended tv show/movies

# In[ ]:


def recnflix (name, n, types):
    if name.lower() in list(dfr2['title2']):
        fav_index = dfr2[dfr2['title2']==name.lower()].index[0]
        n=int(n)
        types = types.lower()
        rec = sorted(list(enumerate(score[fav_index])), key=lambda x:x[1], reverse=True)
        index_rec = []
        i= 1
        while len(index_rec) < n:
            if dfr2.iloc[rec[i][0]]['type'] == types:
                index_rec.append(rec[i][0])
            i += 1
        return df.iloc[index_rec]
    else: 
        return ('The Title You Entered is Wrong')


# <b>Calling the Function</b> (for example)

# In[ ]:


recnflix('Saudi Arabia Uncovered', 3, 'TV Show')


# In[ ]:


recnflix('The Outsider', 4, 'Movie')


# In[ ]:




