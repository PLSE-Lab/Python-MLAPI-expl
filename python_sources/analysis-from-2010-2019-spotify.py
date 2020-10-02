#!/usr/bin/env python
# coding: utf-8

# # Analysis of top songs 2010-2019 Spotify did by the team!
# 
# This analysis was made by 3 Informatics Engineers namely:
# 
# 1. [Filipe Monteiro](https://www.linkedin.com/in/pimonteiro/)
# 2. [Filipa Parente](https://www.linkedin.com/in/filipa-parente-54abb81a1/)
# 3. [Leonardo Silva](http://www.linkedin.com/in/leonardo-js/)

# ## Importing necessary libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import folium
from folium.plugins import HeatMap


# **Importing CSV**

# In[ ]:


data = pd.read_csv('/kaggle/input/top-spotify-songs-from-20102019-by-year/top10s.csv', encoding='ISO-8859-1')
data.head()


# ## Getting some info about the dataset

# **Dataset has 603 rows and 15 columns**

# In[ ]:


data.shape


# **Information each features of dataset as mean, median, standard deviation and others.**

# In[ ]:


data.describe()


# In[ ]:


data.info()


# In[ ]:


data.count()


# ## To work with dataset

# So we can avoid possible problems later on on the dataset, we will rename the column top genres to top_genres, as well as remove the Unnamed :0 column that correspondes to the index. At last, as the dataset as no NaN values, it is then ready for analysis.

# In[ ]:


data = data.rename(columns={'top genre': 'top_genre'})
data = data.drop('Unnamed: 0', axis=1)
print(data.columns)


# In[ ]:


data.head(2)


# #### To check correlation between dataset's features

# In[ ]:


data.corr()


# In[ ]:


plt.figure(figsize=(7, 6))
sn.heatmap(data.corr(),
            annot = True,
            fmt = '.2f',
            cmap='Blues')
plt.title('Correlation between variables in the Spotify data set')
plt.show()


# The features present correlactiones were:
# 
#     -> Acous with NRGY  -0.56
#     -> Val   with DNCE   0.50
#     -> dB    with NRGY   0.54
#     
# **So we plotted some graphs for visualization relationship between this features**
#     

# In[ ]:



data.plot(x='acous',y='nrgy',kind='scatter', title='Relationship between Energy and Acousticness  ',color='r')
plt.xlabel('Acousticness')
plt.ylabel('Energy')
data.plot(x='nrgy',y='dB',kind='scatter', title='Relationship between Loudness (dB) and Energy',color='b')
plt.xlabel('Energy')
plt.ylabel('Loudness (dB)')
data.plot(x='val',y='dnce',kind='scatter', title='Relationship between Loudness (dB) and Valence',color='g')
plt.xlabel('Valence')
plt.ylabel('Loudness (dB)')


# # Dataset analysis
# 
# For answer the analysis questions:

# 
#  # 1. Who are the ten artists with more music in the dataset?

# In[ ]:


artists = data['artist'].unique()
print("The dataset has {} artists".format (len(artists)))


# In[ ]:


artists = data['artist'].value_counts().reset_index().head(10)
print(artists)


# In[ ]:


plt.figure(figsize=(15,10))
sn.barplot(x='index',y='artist', data=artists)
plt.title("Number of Musics on top score by the 10 Top Artists")


# ##### Below we can observe the distribution of songs per year of each singer!

# In[ ]:


frames = []
topArtists = data['artist'].value_counts().head(10).index
for i in topArtists:
     frames.append(data[data['artist'] == i])
        
resultArtist = pd.concat(frames)
artistsYear = pd.crosstab(resultArtist["artist"],resultArtist["year"],margins=False)
artistsYear


# In[ ]:


plt.figure(figsize=(20,10))
for i in artists['index']:
  tmp = []
  for y in range(2010,2020):
    songs = data[data['year'] == y][data['artist'] == i]
    tmp.append(songs.shape[0])
  sn.lineplot(x=list(range(2010,2020)),y=tmp)
plt.legend(list(artists['index']))
plt.title("Evolution of each Top 10 Artists throught the target Years")


# In[ ]:


data['artist'].value_counts().head(10).plot.pie(figsize=(8,8), autopct='%1.0f%%')


# ### Conclusion about this question is:
# 
# Looking into the previous graphs we see that Justin Bieber, even tho is on the top 10 artists with 16 songs he only got the placement thanks to a huge spike between 2014 and 2016, having the rest of the time between 0 and 2 popular songs. On a similar way, Lady Gaga also is a good target to analyse: starting strong in 2010-2011 with 3 to 5 songs making huge sucess, she lost her popularity until 2018-2019 where she got a huge increase on popular songs. This makes sense, since at the end of 2018 she became known for a song on the movie "A Star is Born":

# In[ ]:


data[data['artist'] == 'Lady Gaga'][data['year'] == 2018]


# # 2. Which songs appear more than once and other informations about songs?

# In[ ]:


data['title'].value_counts().head(20)>1


# In[ ]:


plt.figure(figsize=(15,10))
sn.countplot(y=data.title, order=pd.value_counts(data.title).iloc[:19].index, data=data)
topMusics = data['title'].value_counts().head(19).index
plt.title("The Songs appear more than once")


#  #### We ask what the appear distribution each song along years of the dataset

# In[ ]:


plt.figure(figsize=(20,10))
for i in topMusics:
  tmp = []
  for y in range(2010,2020):
    songs = data[data['year'] == y][data['title'] == i]
    tmp.append(songs.shape[0])
  sn.lineplot(x=list(range(2010,2020)),y=tmp)
plt.legend(list(topMusics))
plt.title("Evolution of each Top 10 Artists throught the target Years")


# #### Interessant :
# this case is the song "sugar" of Maroon 5 that appear twice in the same year.

# In[ ]:


data[data['title']== 'Sugar']


# ### Curiosities about songs: 
# 
# ##### Top 15 songs by Popularity: 
# 
# Conclusion that 2019 was a year with the most popular songs, with almost all songs being pop!

# In[ ]:


data.sort_values(by=['pop'], ascending=False).head(15)


# #### Curiosity about songs: The Top 15 longer songs:

# In[ ]:


data.sort_values(by=['dur'], ascending=False).head(15)


# #### Curiosity about songs: Top 15 songs by Acousticness:

# In[ ]:


data.sort_values(by=['acous'], ascending=False).head(15)


# # 3. Which of the ten genders is more popular between the dataset?

# In[ ]:


genres = data['top_genre'].value_counts().reset_index().head(10)


# In[ ]:


plt.figure(figsize=(23,10))
sn.barplot(x='index',y='top_genre', data=genres)


# In[ ]:


data['top_genre'].value_counts().head(10).plot.pie(figsize=(8,8), autopct='%1.0f%%')


# In[ ]:


plt.figure(figsize=(20,10))
for i in genres['index']:
  tmp = []
  for y in range(2010,2020):
    songs = data[data['year'] == y][data['top_genre'] == i]
    tmp.append(songs.shape[0])
  sn.lineplot(x=list(range(2010,2020)),y=tmp)
plt.legend(list(genres['index']))


# #### We think about the question with analysis data were:
# Here we can see how pop music is popular in your generation. What if we looked back into the '80s, '90s, would we see a big difference? Type of songs per top artist.

#    # 4. Where did the top 10 artists come from?

# In[ ]:


artists


# #### The original idea was to take where is from the artist through the API, but we still can't do it!
# put hand on!

# In[ ]:


dicArtists = {
    'Katy Perry':"Santa Barbara",
    'Justin Bieber':"London Canada",
     'Rihanna':"Saint Michael",
    'Maroon 5':"Los Angeles",
    'Lady Gaga':"Manhattan",
    'Bruno Mars':"Honolulu", 
    'The Chainsmokers':"Times Square" ,
    'Pitbull':"Miami",
    'Shawn Mendes':"Toronto",
    'Ed Sheeran':"United Kingdom", 
  }


# We will need coordinates, but we won't put hand on, because we will use the library to make it for us. We put one place and she gives coordinates.

# In[ ]:


#!pip install folium
get_ipython().system('pip install geocoder')


# In[ ]:


import geocoder
listGeo = []

for value in (dicArtists.values()):
    g = geocoder.arcgis(value)
    listGeo.append(g.latlng)


# In[ ]:


top_genres =[]
for key in (dicArtists.keys()):
    top_genres.append(data[data['artist']== key].top_genre.unique())


# In[ ]:


lat = []
log = []
for i in listGeo:
    lat.append(i[0])
    log.append(i[1])


# In[ ]:


colors = {
 'dance pop': 'pink',
 'pop': 'blue',
 'barbadian pop': 'green',
 'electropop': 'orange',
 'canadian pop': 'red',
}
        


# In[ ]:


dfLocation = pd.DataFrame(columns=['Name','Lat','Log','Gen'])
dfLocation['Name'] = artists['index']
dfLocation['Gen']  = np.array(top_genres)
dfLocation['Lat']  = lat
dfLocation['Log']  = log
dfLocation


# ### plotting the empty globe graph

# In[ ]:


spotify = folium.Map(
    location=[41.5503200,-8.4200500],# Coordenadas retiradas do Google Maps
    zoom_start=2
)
spotify


# #### plotting the globe chart with the artists' origins

# In[ ]:


for i in range(10):
    singer = dfLocation.iloc[i]
    folium.Marker(
        
        popup=singer['Name']+'-'+singer['Gen'],
        location=[singer['Lat'], singer['Log']],
    icon=folium.Icon(color=colors[singer['Gen']], icon='music')).add_to(spotify)
    
spotify


# You can click on the music symbol and see information such as: artist name and genre.
# In New York has two artists 

# ### plotting the globe graph with heat map

# See the heat map with the top 10

# In[ ]:


spotify = folium.Map(
    location=[41.5503200,-8.4200500],# Coordenadas retiradas do Google Maps
    zoom_start=2
)

HeatMap(list(zip(lat, log))).add_to(spotify)
spotify


# Let's see the heat map with the top 20 artists

# In[ ]:


dic = {
    'Katy Perry':"Santa Barbara",
    'Justin Bieber':"London Canada",
     'Rihanna':"Saint Michael",
    'Maroon 5':"Los Angeles",
    'Lady Gaga':"Manhattan",
    'Bruno Mars':"Honolulu", 
    'The Chainsmokers':"Times Square" ,
    'Pitbull':"Miami",
    'Shawn Mendes':"Toronto",
    'Ed Sheeran':"United Kingdom", 
    'Jennifer Lopez':'Castle Hill',
    'Calvin Harris' :  'Dumfries',  
    'Adele'  : 'Tottenham',
    'Kesha'     :  'California',
    'Justin Timberlake'   : 'Memphis' ,  
    'David Guetta '      :'Paris',
    'OneRepublic'       :'Colorado',
    'Britney Spears '    : 'Mississippi',
    'Ariana Grande '      :'Florida',
    'Taylor Swift'       :'Pennsylvania',  
  }
listGeo = []
for value in (dic.values()):
    g = geocoder.arcgis(value)
    listGeo.append(g.latlng)

lat = []
log = []
for i in listGeo:
    lat.append(i[0])
    log.append(i[1])

spotify = folium.Map(
    location=[41.5503200,-8.4200500],# Coordenadas retiradas do Google Maps
    zoom_start=2
)

HeatMap(list(zip(lat, log))).add_to(spotify)
spotify


# # We try to use a grouping algorithm, K-means, to group the songs by genre according to their characteristics.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re, os
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# **Some basical information from dataset**

# In[ ]:


len(data['top_genre'].unique())


# #### This dataset has 50 genres. 

# # K Means Cluster Creation
# 
# Now it is time to create the Cluster labels!
# 
# ** Import KMeans from SciKit Learn.**

# In[ ]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder


# #### Transform categorical variables into numerical variables

# In[ ]:


labelenconder = LabelEncoder()
data['title'] = labelenconder.fit_transform(data['title'].astype('str'))
data['artist'] = labelenconder.fit_transform(data['artist'].astype('str'))
target = data['top_genre']


# In[ ]:


train = data.drop(columns=['top_genre'], axis=1) 
train.head()


# ## Elbow Method
# In order to get the best K to apply on our cluster, we'll apply Elbow Method

# In[ ]:


Sum_of_squared_distances = []
std = StandardScaler()
std.fit(train)
data_transformed = std.transform(train)

K = range(1,60)
for i in K:
    km = KMeans(n_clusters=i)
    km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)


# In[ ]:


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# ##### Using elbow method to determine the best number of clusters, the best case rounds between 40 and 50.

# ## KMeans Model
# 
# However, as we want to determine if a song is pop genre or not, we use value 2 on our model
# Create an instance of K Means model with 2 clusters

# In[ ]:


km = KMeans(n_clusters=2)
km.fit(train)
km.cluster_centers_


# ## Evaluation
# 
# Now let's define a converter that defines all pop genres (neo, canadian, etc.) as 1 and 0 for other genres

# In[ ]:


def converter(cluster):
    result = re.findall(".*pop",cluster)
    if len(result) != 0:
        return 1
    else:
        return 0


# #### Then, we apply the converter and we add a new column called 'is pop' 

# In[ ]:


data['is_pop'] = data['top_genre'].apply(converter)
data.head()


# #### Now we create a confusion matrix and a classification report, in order to see how well the Kmeans clustering worked without any labels. 

# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(data['is_pop'],km.labels_))
print(classification_report(data['is_pop'],km.labels_))


# #### From here we came to the conclusion that this model isn't a good cluster of pop/non pop songs 
# 
# #### However, we can see that 80% of top songs are pop ones

# ## KMeans Model with cross validation
# Another approach is separate our dataset using train_test_split

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train,data['is_pop'],
                                                    test_size=0.30)


# In[ ]:


km.fit(X_train,y_train)
X_test.head()


# In[ ]:


pred = km.predict(X_test)
print("Confusion Matrix \n")
print(confusion_matrix(y_test,pred))
print("\n Metrics: \n \n")
print(classification_report(y_test,pred))


# #### As we can observe this approach is equal than the other applied above, with bad accuracy (50%)
# 
# #### From here we came to the conclusion that both approaches don't seem to be good to build a cluster of pop/non pop songs 

# ![Alt Text](https://media.giphy.com/media/3ohs7JG6cq7EWesFcQ/giphy.gif)
