#!/usr/bin/env python
# coding: utf-8

# Author: Shruti Gupta <br>
# Date: 22/02/2020 <br>
# Version: 2

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats
import squarify as sq
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import sklearn
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score, KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from wordcloud import WordCloud, STOPWORDS
get_ipython().run_line_magic('matplotlib', 'inline')


# Any results you write to the current directory are saved as output.


# Date: 05/01/2020<br>
# Version: 1<br>
# **Checking of top 50 of songs and artist in the year 2019 on Spotify.**

# In[ ]:


#reading the data
music = pd.read_csv('../input/top50spotify2019/top50.csv',encoding='ISO-8859-1')


# getting the initial discription of top 50 songs from spotify

# In[ ]:


music.shape


# In[ ]:


music.info()


# In[ ]:


music.head(10)


# In[ ]:


music.describe()


# dropping unwanted columns and filling any missing values

# In[ ]:


music.drop('Unnamed: 0', axis=1, inplace=True)


# In[ ]:


music.isnull().sum()
music=music.fillna(0)


# In[ ]:


music.isnull()


# Displaying intital graphs

# In[ ]:


music.hist(figsize=(20,20))


# In[ ]:


music['Genre'].value_counts().plot.bar()
plt.suptitle('Counts for Genre')


# In[ ]:


music_corr = music.corr()
sns.heatmap(music_corr, annot=True,cmap ='RdYlGn')
plt.suptitle('Correlation for Top 50')


# Further Calculations

# In[ ]:


#changing columns
music.columns = ['Track', 'Artist', 'Genre', 'BPM', 'Energy', 'Danceability', 'Loudness', 
               'Liveness', 'Valence', 'Length', 'Acousticness', 'Speechiness', 'Popularity']


# In[ ]:


#Most popular artist
art_pop = music.groupby('Artist')['Popularity'].mean().sort_values(ascending = False)
print("Artist Popularity")
art_pop


# In[ ]:


#Finding out the skew for each attribute
skew=music.skew()
print(skew)
# Removing the skew by using the boxcox transformations
transform=np.asarray(music[['Liveness']].values)
music_transform = stats.boxcox(transform)[0]
# Plotting a histogram to show the difference 
plt.hist(music['Liveness'],bins=10)#original data
plt.suptitle('Skewness for Music Liveness')
plt.show()
plt.hist(music_transform,bins=10) #corrected skew data
plt.suptitle('Skewness for Music Liveness after Transformation')
plt.show()


# Word Cloud

# In[ ]:


artist_list = music["Artist"].unique().tolist()
artist_list[:2]


# In[ ]:


artist = " ".join(artist_list)
artist[:100]


# In[ ]:


# create a word cloud for artist
artist_wordcloud = WordCloud().generate(artist)
# show the created image of word cloud
plt.figure()
plt.imshow(artist_wordcloud)
plt.show()


# In[ ]:


#word cloud for genre
genre_list = music["Genre"].unique().tolist()
genre_list[:2]
genre = " ".join(genre_list)
genre[:100]
# create a word cloud for artist
genre_wordcloud = WordCloud().generate(genre)
# show the created image of word cloud
plt.figure()
plt.imshow(genre_wordcloud)
plt.show()


# Some Machine Learning to Aanalysis Spotify Top 50

# In[ ]:


le = LabelEncoder()

for col in music.columns.values:
  if music[col].dtypes == 'object':
    le.fit(music[col].values)
    music[col] = le.transform(music[col])

music.head()


# In[ ]:


# Create test and train dataset
X = music.drop('Loudness', axis=1)
y = music.Loudness


# In[ ]:


X.drop('Artist', axis=1, inplace=True)

# Creating a test and training dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Scaling the data
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

type(X_train), type(X_test)


# In[ ]:


# Linear Regression
Lin_reg = LinearRegression()
Lin_reg.fit(X_train, y_train)

y_pred = Lin_reg.predict(X_test)

print(Lin_reg.intercept_, Lin_reg.coef_)
print(mean_squared_error(y_test, y_pred))


# In[ ]:


#SVR
SVR_Reg = SVR(C=0.5)
SVR_Reg.fit(X_train, y_train)

y_pred = SVR_Reg.predict(X_test)
print(mean_squared_error(y_test, y_pred))


# In[ ]:


#KMEAN
clustering = KMeans(n_clusters=2)
clustering.fit(X_train, y_train)

y_pred = clustering.predict(X_test)
print(mean_squared_error(y_test, y_pred))


# Based on this it seens that SVR best calculations
