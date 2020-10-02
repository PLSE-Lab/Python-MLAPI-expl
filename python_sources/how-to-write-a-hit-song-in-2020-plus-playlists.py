#!/usr/bin/env python
# coding: utf-8

# # Spotify Top Tracks of the Decade

# ##  In this exercise I decided to use a data set from Spotify that contained the top 50 songs for each year from 2010-2019.  As a musician I was curious to see what a hit song in 2020 might look like (and how to write one!), and also to test my hypothesis that popular songs are getting more sad sounding.  Then for fun I decided to see if I could use Machine Learning to make playlists based off of their feature characteristics.

# In[ ]:


# Data set can be found here!

#                           https://www.kaggle.com/leonardopena/top-spotify-songs-from-20102019-by-year


# ### Import packages and read data into a dataframe

# In[ ]:


#first import of packages 

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns

import sklearn
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import classification_report, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:



df = pd.read_csv("../input/top-spotify-songs-from-20102019-by-year/top10s.csv", encoding='ISO-8859-1') 



df.head()


# In[ ]:


#Renaming the columns

df.rename(columns={'title':'Track Name','artist':'Artist Name','bpm':'Beats Per Minute','top genre':'Genre','nrgy':'Energy','dnce':'Danceability', 'dB':'Loudness dB','spch':'Speechiness','live':'Liveness','val':'Valence','dur':'Length','acous':'Acousticness','pop':'Popularity'},inplace=True)
df.head()


# # Part 1 - EDA and Data Visualization

# In[ ]:


# The datatypes of the different columns

print(df.dtypes)


# In[ ]:


# Get initial descriptive statistics on the columns

pd.set_option('precision', 3)
df.describe()


# ## Categorical EDA with bar graphs to see most popular Artists, Genres, and Tracks

# In[ ]:


#Calculating the number of songs of each genre

print(type(df['Genre']))

popular_genre = df.groupby('Genre').size()

popular_genre = popular_genre.sort_values(ascending=False)

popular_genre

genre_list = df['Genre'].values.tolist()

genre_top20 = popular_genre[0:20,]

genre_top20 = genre_top20.sort_values(ascending=True)

genre_top20 = pd.DataFrame(genre_top20, columns = [ 'Number of Songs'])

genre_top20.head()


# In[ ]:


plt.figure(figsize=(16,8))


ax = sns.barplot(x = 'Number of Songs' , y = genre_top20.index , data = genre_top20, orient = 'h', palette = sns.color_palette("muted", 20), saturation = 0.8)

plt.title("Top 20 Genres of the Decade... That's a lot of Pop!",fontsize=30)
plt.xlabel('Number of Songs', fontsize=25)
plt.ylabel('Genre', fontsize=10)

xticks = [0, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340]

plt.xticks(xticks, size=20,rotation=45)
plt.yticks(size=20)
sns.despine(bottom=True, left=True)


plt.show


# In[ ]:


#Pie chart to show top 20 genres

labels = genre_top20.index
sizes = genre_top20.values

explode = (  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1)

plt.figure(figsize = (10,10))

plt.pie(sizes, labels = labels, explode = explode)

plt.title("Top 20 Genres on the List", fontsize=16)

autopct=('%1.1f%%')
plt.axis('equal')

plt.show()


# In[ ]:


#Calculating the least popular genres


genre_bot29 = popular_genre[21:,]

genre_bot29 = genre_bot29.sort_values(ascending=True)

genre_bot29 = pd.DataFrame(genre_bot29, columns = [ 'Number of Songs'])

genre_bot29.head()


# In[ ]:


#Pie chart to show bottom 35 genres

labels = genre_bot29.index
sizes = genre_bot29.values

explode = ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1)

plt.figure(figsize = (10,10))

plt.pie(sizes, labels = labels, explode = explode)

plt.title("Least Popular Genres on the Top 50 List", fontsize=16)

autopct=('%1.1f%%')
plt.axis('equal')

plt.show()


# In[ ]:


#Calculating the number of songs by each of the artists


popular_artist = df.groupby('Artist Name').size()

popular_artist = popular_artist.sort_values(ascending=False)

popular_artist

artist_list=df['Artist Name'].values.tolist()

artist_top25 = popular_artist[0:25,]

artist_top25 = artist_top25.sort_values(ascending=True)

artist_top25 = pd.DataFrame(artist_top25, columns = [ 'Number of Songs'])

artist_top25.head() 


# In[ ]:


plt.figure(figsize=(16,8))


ax = sns.barplot(x = 'Number of Songs' , y = artist_top25.index , data = artist_top25, orient = 'h', palette = sns.color_palette("muted", 25), saturation = 0.8)

plt.title("Top 25 Artists of the Decade",fontsize=30)
plt.xlabel('Number of Songs on Top 50 List', fontsize=25)
plt.ylabel('Artist', fontsize=10)

xticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

plt.xticks(xticks, size=20,rotation=45)
plt.yticks(size=20)
sns.despine(bottom=True, left=True)


plt.show


# In[ ]:


# Songs that made the top 50 list twice

plt.figure(figsize=(16,8))

tracks = pd.value_counts(df['Track Name']).iloc[:18].index

sns.countplot(df['Track Name'], order = tracks, orient = 'h', palette = sns.color_palette("magma", 25), saturation =0.7)

plt.title('Songs That Made the Top 50 List on Two Different Years',fontsize=30)
plt.xlabel('Track', fontsize=25)
plt.ylabel('Number of Years in Top 50 List', fontsize=25)

plt.xticks(size=20,rotation=90)
plt.yticks( [0, 1, 2]  , size=20)
sns.despine(bottom=True, left=True)


plt.show


# ## Investigating  bad data

# In[ ]:


# Plotting a histogram to show the spread of Popularity since we notice some strange stats worth investigating

plt.hist(df['Popularity'],bins=100)

plt.show()


# In[ ]:


# Investigating low popularity

low_pop = df[df['Popularity'] <= 20]

low_pop


# In[ ]:


# Inspect bad data...  How can the popularity be 0 if these are top 50 songs?

df.loc[df['Popularity']==0]


# In[ ]:


# drop bad data

df = df.drop(df.index[[50, 138, 267, 362, 442]])

df = df.reset_index()

# check it's gone

df.iloc[[50, 138, 267, 362, 442]]


# In[ ]:


# clean up index

df = df.drop('index', axis=1)
df = df.drop('Unnamed: 0', axis=1)


df.head()


# # Part 2 - Statistical EDA, Normalization, and PCA

# In[ ]:


# Get descriptive statistics on the columns to see the change

pd.set_option('precision', 3)

df.describe()


# In[ ]:


# get df ready for scatter matrix

df_features = df.drop(df.columns[[0, 1, 2, 3]], axis =1)

df_features.head()


# ## Normalization of the data to get the values between 0 and 1 in order to help with  PCA and regression analysis

# In[ ]:


# Normalize the data with Min/Max

df_norm = df_features

scaler = MinMaxScaler() 

df_norm = scaler.fit_transform(df_norm)

df_norm = pd.DataFrame(df_norm, columns = df_features.columns)

df_norm.describe()


# In[ ]:


# Plotting a histogram to show the difference (note the x-axis)

plt.hist(df_features['Loudness dB'], bins=10)     #original data
plt.show()


plt.hist(df_norm['Loudness dB'], bins=10)          #standardized data
plt.show()


# ## PCA (Principle Component Analysis) to reduce the feature columns

# In[ ]:


#Fitting the PCA algorithm with our Data

pca = PCA().fit(df_norm)


# In[ ]:


#Plotting the Cumulative Summation of the Explained Variance

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Spotify Data Explained Variance')
plt.show()


# In[ ]:


# print the explained variance for each component

explained_variance = pca.explained_variance_ratio_

print(explained_variance)


# In[ ]:


# how much variance can be explained for 8 components

print('The explained variance for this many components is:  ',explained_variance[0:8].sum())


# In[ ]:


# visually inspect pca

map = pd.DataFrame(pca.components_, columns=df_norm.columns)
plt.figure(figsize=(12,6))
sns.heatmap(map, cmap='gist_earth_r')


# In[ ]:


# choose number of components

pca = PCA(n_components = 8)

data_pca = pca.fit_transform(df_norm)


# ## Analyzing the features and their interdependencies

# In[ ]:


# Visualizing the relationship between all features

scatter_matrix(df_norm)

plt.gcf().set_size_inches(30, 30)

plt.show()


# In[ ]:


# Use a spearman correlation to measure the relationship between features

pd.set_option('display.width', 100)
pd.set_option('precision', 3)

correlation = df_norm.corr(method='spearman')

print(correlation)


# In[ ]:


# heatmap of the correlation to visualize the relationships between features

plt.figure(figsize=(10,10))
plt.title('Correlation heatmap')

sns.heatmap(correlation, annot = True, vmin=-1, vmax=1, cmap="YlGnBu", center=1)


# In[ ]:


# Analysing the relationship between Danceablity and Valence

fig = plt.subplots(figsize = (10,10))

sns.regplot(x = 'Valence', y = 'Danceability', data = df_norm, color = 'olive')

sns.kdeplot(df_norm['Valence'], df_norm['Danceability'])

print('The spearman correlation is:  ',correlation['Danceability']['Valence'])


# ###  _**Happier songs are more danceable!**_

# In[ ]:


# Analysing the relationship between valence vs popularity


f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True, start=2.8, rot=.1)

sns.kdeplot(df['Valence'], df['Popularity'], cmap=cmap, n_levels=16, shade=True);


# ###  _**My hypothesis that popular songs are more depressing was wrong!**_

# In[ ]:


# Analysing the relationship between valence vs energy

sns.jointplot(x=df['Valence'], y=df['Energy'], data=df, kind="kde", color='lightblue');


# ###  _**Sadder songs do have less energy though!**_

# In[ ]:


# Analysing the relationship between valence vs length


sns.jointplot(df['Valence'], df['Length'], kind="hex", color="#4CB391")


# In[ ]:


# Analysing the relationship between loudness vs danceability

sns.catplot(y="Danceability", x="Loudness dB", kind = "swarm", data = df_features, palette = 'rocket_r')


# In[ ]:


# Analysing the spread of popularity throught the years

sns.catplot(y = "Popularity", x = "year", kind = "box", data = df, palette = 'seismic')


# ###  _**The most recent songs are the most popular!  ...this makes me question the Popularity score**_

# ## Analyzing the trends over the years

# In[ ]:


# PairGrid to analyze trends over the years

sns.set()

g = sns.PairGrid(df, y_vars = ['Beats Per Minute', 'Energy', 'Danceability', 'Loudness dB', 
                               'Liveness', 'Valence', 'Length', 'Acousticness', 'Speechiness', 'Popularity'] , x_vars = ['year'], aspect = 4)

g = g.map(sns.lineplot, color="blue")

# Adjust the tick positions and labels

g.set(xticks=[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020])


# Adjust the arrangement of the plots

g.fig.subplots_adjust(wspace=.02, hspace=.02);


# ###  _**In general hit songs are getting slower, less energy, way more danceable, a little louder, slighly happier, way shorter, way more acoustic, with less lyrics!**_

# # Part 3 - Multiple Linear Regression to make a prediction for 2020

# In[ ]:


# insert year cloumn into features df

df_features.insert(0, 'year', df['year'])


df_features.head()


# ## Choose the Best Multiple Linear Regression Model

# In[ ]:


# setup features and target


X = df_features[['year']]
y = data_pca

# can switch variable z to y to see effect of all features on predicition (also change y to z)
z = df_features[['Beats Per Minute', 'Energy', 'Danceability', 'Loudness dB', 'Liveness', 'Valence', 'Length', 'Acousticness', 'Speechiness', 'Popularity']]


# In[ ]:


# Train, Test, Split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# Instantiate model
mlr = LinearRegression()

# Fit Model
mlr.fit(X_train, y_train)

# Predict
y_pred = mlr.predict(X_test)


# RMSE
print('The Root Mean Squared Error for this model is:  ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:


# K-Fold Cross Val 

mlr = LinearRegression()


mlr.fit(X, y)


mse = cross_val_score(mlr, X, y, scoring='neg_mean_squared_error', cv=10)


# fix the sign of MSE scores
mse_scores = -mse


# convert from MSE to RMSE
rmse_scores = np.sqrt(mse_scores)


# calculate the average RMSE
print('The Root Mean Squared Error for this model is:  ', rmse_scores.mean())


# In[ ]:


# Ridge Regression and GridSearchCV

ridge = Ridge()

params = { 'alpha' : [ 1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 40, 80, 100, 1000, 10000 ]  }

rr = GridSearchCV(ridge, params, scoring = 'neg_mean_squared_error', cv=10)

rr.fit(X, y)

print(rr.best_params_)
print(rr.best_score_)

rr_mse = -(rr.best_score_)

rr_rmse = np.sqrt(rr_mse)

print('The Root Mean Squared Error for this model is:  ', rr_rmse)


# In[ ]:


# Lasso Regression and GridSearchCV

lasso = Lasso()

params = { 'alpha' : [ 1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 40, 80, 100, 1000, 10000 ]  }

lr = GridSearchCV(lasso, params, scoring = 'neg_mean_squared_error', cv=10)

lr.fit(X, y)

print(lr.best_params_)
print(lr.best_score_)

lr_mse = -(lr.best_score_)

lr_rmse = np.sqrt(lr_mse)

print('The Root Mean Squared Error for this model is:  ', lr_rmse)


# ### _**It looks like the K-Fold Cross Validation MLR model is the best!**_

# ## Predicition

# In[ ]:


# predict a hit song in 2020's features

hit = mlr.predict([[2020]])

hit


# In[ ]:


# reverse pca

hit = pca.inverse_transform(hit)

hit


# In[ ]:


# reverse normalization

hit = scaler.inverse_transform(hit)
    
hit


# ## Finding a song from our data to serve as an exemplar for the predicted values by using the machine learning technique, K-Nearest Neighbor Classifier

# In[ ]:


# get the features of the prediciton into a dataframe

hit = pd.DataFrame(hit)

hit = hit.drop(columns = 9, axis=1)

hit


# In[ ]:


# make a prediction for 2020 using the machine learning classifier KNN


knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(df_features[['Beats Per Minute','Energy','Danceability','Loudness dB','Liveness','Valence', 'Length', 'Acousticness', 'Speechiness']], df_features.index)

y_pred = knn.predict(hit)

y_pred = pd.DataFrame(y_pred)

y_pred


# In[ ]:


# look up the index

winner = df.iloc[[388]]

winner


# #  _**There you have it... If you wish to write a hit song in 2020 you can write one just like Close by Nick Jonas !**_  

# ##  _**And again, in 2020 hit songs are getting slower, less energy, way more danceable, a little louder, slightly happier, way shorter, way more acoustic, with less lyrics!**_

# # Part 4 - Make Playlists based off of the feature characteristics using Machine Learning technique K-Means Clustering

# ### Standardization of the data to fix skew and get mean=0 and std=1 in order to help with clustering
# 

# In[ ]:


# Finding out the skew for each feature

skew = df_features.skew()

print(skew)


# In[ ]:


# scale the data

scaler = StandardScaler()

df_scaled = scaler.fit_transform(df_features)

df_scaled = pd.DataFrame(df_scaled)

df_scaled.head()


# In[ ]:


# Plot to show the difference

plt.hist(df_features['Speechiness'], bins=10)                    #original data
plt.show()

plt.hist(df_scaled.iloc[8], bins=10)                            #standardized data
plt.show()


# ## Find the appropriate amount of clusters
# 

# In[ ]:


# choose the best number of clusters using elbow method and inertia

k = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

inertias = []

for i in k:
    km = KMeans(n_clusters=i, max_iter=1000, random_state=42)
    km.fit(df_scaled)
    inertias.append(km.inertia_)

plt.plot(k, inertias)
plt.xlabel("Value for k")
plt.ylabel("Inertias")
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

plt.show()


# In[ ]:


k = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

score=[]

for n_cluster in k:
    kmeans = KMeans(n_clusters=n_cluster).fit(df_scaled)
    silhouette_avg = silhouette_score(df_scaled, kmeans.labels_)
    score.append(silhouette_score(df_scaled, kmeans.labels_))
    
    print('Silhouette Score for %i Clusters: %0.4f' % (n_cluster, silhouette_avg))


# In[ ]:


# plot cluster options

plt.plot(k, score, 'o-')
plt.xlabel("Value for k")
plt.ylabel("Silhouette score")
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

plt.show()


# In[ ]:


# set number of clusters

kclusters = 8


# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, init='k-means++', random_state=42).fit(df_scaled)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# In[ ]:


# add clustering labels to dataframe

df.insert(0, 'Playlist Number', kmeans.labels_)

df.head()    # check out the Cluster Labels column!


# ## Playlist #1

# In[ ]:


df.loc[df['Playlist Number'] == 0, df.columns[[1, 2]]]


# ## Playlist #2

# In[ ]:


df.loc[df['Playlist Number'] == 1, df.columns[[1, 2]]]


# ## Playlist #3

# In[ ]:


df.loc[df['Playlist Number'] == 2, df.columns[[1, 2]]]


# ## Playlist #4

# In[ ]:


df.loc[df['Playlist Number'] == 3, df.columns[[1, 2]]]


# ## Playlist #5

# In[ ]:


df.loc[df['Playlist Number'] == 4, df.columns[[1, 2]]]


# # Playlist #6

# In[ ]:


df.loc[df['Playlist Number'] == 5, df.columns[[1, 2]]]


# # Playlist #7

# In[ ]:


df.loc[df['Playlist Number'] == 6, df.columns[[1, 2]]]


# # Playlist #8

# In[ ]:


df.loc[df['Playlist Number'] == 7, df.columns[[1, 2]]]


# # Now all we have to do is name these playlists... I'll leave that up to you!

# In[ ]:




