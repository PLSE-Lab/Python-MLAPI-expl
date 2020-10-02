#!/usr/bin/env python
# coding: utf-8

# Movies Recommendation. The main idea behind this solution is to predict a movie's rating by a particular user, based on the ratings it has given to other films and also based on the ratings of other users of that movie.
# Also, to make not random selection of films, I created a recommendation system that, based on movie data (genres, actors, directors, keywords), picks similar ones.

# Thanks to:
# * https://www.kaggle.com/fabiendaniel/film-recommendation-engine
# * https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system
# * https://www.kaggle.com/sjj118/movie-visualization-recommendation-prediction

# Connecting libraries

# In[ ]:


import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD, KNNBaseline
from surprise.model_selection import cross_validate
import seaborn as sn
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import plotly.graph_objs as go 
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime
from fbprophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns


# Downloading data

# In[ ]:


credits = pd.read_csv('../input/the-movies-dataset/credits.csv')
moviesMetaData = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv',low_memory=False)
keywords = pd.read_csv('../input/the-movies-dataset/keywords.csv')
ratings = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')


# In[ ]:


ratings.rename(columns={'movieId': 'id'}, inplace = True)


# We change the column type 'id' in all dateframes so that they can be combined

# In[ ]:


moviesMetaData['id'] = moviesMetaData['id'].astype(str)
credits['id'] = credits['id'].astype(str)
keywords['id'] = keywords['id'].astype(str)
ratings['id'] = ratings['id'].astype(str)


# In[ ]:


ratings = ratings[['id']]
ratings = ratings.drop_duplicates()

moviesMetaData = pd.merge(moviesMetaData,ratings, on='id')


# We merge the dateframes into one, and select only those columns to use for the recommendation.

# In[ ]:


mainList= pd.merge(moviesMetaData, credits, on='id')
mainList= pd.merge(mainList,keywords, on='id')
corrMatrix = mainList.corr()
sn.heatmap(mainList.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# In[ ]:


def missingDF(data):
    missing_df = data.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df['filling_factor'] = (mainList.shape[0] 
                                    - missing_df['missing_count']) / mainList.shape[0] * 100
    missing_df.sort_values('filling_factor').reset_index(drop = True)
    
    missing_df = missing_df.sort_values('filling_factor').reset_index(drop = True)
    y_axis = missing_df['filling_factor'] 
    x_label = missing_df['column_name']
    x_axis = missing_df.index

    fig = plt.figure(figsize=(11, 4))
    plt.xticks(rotation=80, fontsize = 14)
    plt.yticks(fontsize = 13)

    plt.xticks(x_axis, x_label,family='fantasy', fontsize = 14 )
    plt.ylabel('Filling factor (%)', family='fantasy', fontsize = 16)
    plt.bar(x_axis, y_axis);
    
    return missing_df


# In[ ]:


table = missingDF(mainList)


# In[ ]:


mainList['release_date'] =  pd.to_datetime(mainList['release_date']) 
mainList['years'] = mainList['release_date'].apply(lambda x: x.year)

mainList[(mainList['years'] < 2019) & (mainList['years'] >= 1950)].groupby(by = 'years').mean()['vote_count'].plot()


# In[ ]:


mainList['budget'] = mainList['budget'].astype(float)
mainList['popularity'] = mainList['popularity'].astype(float)
sn.heatmap(mainList.corr(), annot=True)


# In[ ]:


mainList = mainList[['id', 'title', 'cast', 'crew', 'keywords', 'genres']]


# In[ ]:


mainList.head(5)


# In[ ]:



features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    mainList[feature] = mainList[feature].apply(literal_eval)
    
mainList.head()


# Function for finding the director in 'crew' data, if the director is not returned NaN

# In[ ]:


def getDirector(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# The function to find the first three elements, if the elements are less than 3, return all that is, if there are no elements - an empty list.

# In[ ]:


def getFirstThree(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        
        if len(names) > 3:
            names = names[:3]
        return names

    return []


# Data filtering, we choose only the ones we need. The 'getFirstThree' function is applied to the following data: cast, keywords and genres.

# In[ ]:


mainList['director'] = mainList['crew'].apply(getDirector)

features = ['cast', 'keywords', 'genres']
for feature in features:
    mainList[feature] = mainList[feature].apply(getFirstThree)


# In[ ]:


mainList[['title', 'cast', 'director', 'keywords', 'genres']].head(3)


# In[ ]:


def counting_values(df, column):
    value_count = {}
    for row in df[column].dropna():
        if len(row) > 0:
            for key in row:
                if key in value_count:
                    value_count[key] += 1
                else:
                    value_count[key] = 1
        else:
            pass
    return value_count


# In[ ]:


def count_director(df, column):
    value_count = {}
    for key in df[column].dropna():
        if key in value_count:
            value_count[key] += 1
        else:
            value_count[key] = 1
        
    return value_count
    


# In[ ]:


sn.heatmap(mainList.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# In[ ]:


def hist(data):
    iData = dict(sorted(data.items(), key=lambda x: x[1],reverse=True)[:20])
    pos = np.arange(len(iData.keys()))
    width = 1.0
    
    ax = plt.axes()
    ax.set_xticks(pos + (width / 2))
    ax.set_xticklabels(iData.keys())
    plt.yticks(fontsize = 15)
    plt.xticks(rotation=85, fontsize = 15)
    plt.grid()
    plt.bar(iData.keys(), iData.values(), width, align = 'center', color='g')
    plt.show()


# In[ ]:


def createWordCloud(data):
    wordcloud = WordCloud(max_font_size=100)

    wordcloud.generate_from_frequencies(data)
     
    plt.figure(figsize=[10.1,10.1])
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    plt.show()
    genres_count = pd.Series(data)
    genres_count.sort_values(ascending = False).head(20).plot(kind = 'bar', grid='True')


# In[ ]:


createWordCloud(counting_values(mainList, 'genres'))


# In[ ]:


createWordCloud(counting_values(mainList, 'cast'))


# In[ ]:


createWordCloud(counting_values(mainList, 'keywords'))


# In[ ]:


createWordCloud(count_director(mainList, 'director'))


# Function for deleting spaces between name and surname, actors and director. This is to uniquely identify the actors and directors, rather than a separate name and surname.

# In[ ]:


def deletingSpaces(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
        
features = ['cast', 'keywords', 'director']

for feature in features:
    mainList[feature] = mainList[feature].apply(deletingSpaces)
    
mainList.head(5)


# We combine all the data we use for the recommendation into one row.

# In[ ]:


def combineKeywords(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

mainList['myKeywords'] = mainList.apply(combineKeywords, axis=1)

mainList['myKeywords'].head(5)


# In[ ]:


table = missingDF(mainList)


# With the help of "CountVectorizer" we build a matrix of word occurrence. With "cosine_similarity" we determine the similarity between films.

# In[ ]:


count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(mainList['myKeywords'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)


# We create an array to find the movie index by its name. (For ease of use)

# In[ ]:


mainList = mainList.reset_index()
indices = pd.Series(mainList.index, index=mainList['title'])


# The recommendation feature, based on our selected data, accepts the movie name and returns 10 recommendations.

# In[ ]:


def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]
    
    sim_value = [x[1] for x in sim_scores]
    
    result = indices.iloc[movie_indices]
    
    result[0:10] = sim_value

    return(result)


# In[ ]:


def show(title):
    result = get_recommendations(title)

    plt.figure(figsize=(10,5))
    sn.barplot(x = result[0:10], y=result.index)
    plt.title("Recommended Movies from " + str.upper(title), fontdict= {'fontsize' :20})
    plt.xlabel("Cosine Similarities")
    plt.show()


# In[ ]:


show('Twelve Monkeys')


# In[ ]:


show('Twelve Monkeys')


# In[ ]:


get_recommendations('Twelve Monkeys')


# In[ ]:


reader = Reader()
ratings = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')
corrMatrix = ratings.corr()
ratings.head()


# In[ ]:


corrMatrix = ratings.corr()
sn.heatmap(corrMatrix, annot=True)


# In[ ]:


data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)


# Compared several algorithms:
# 
# SVD - {'test_rmse': array([0.89708382, 0.89360817, 0.90175499, 0.89434609, 0.89912869]),
#  'test_mae': array([0.69280077, 0.6873133 , 0.69288119, 0.68789517, 0.69535891])
#  
# NMF - {'test_rmse': array([0.95434438, 0.95271784, 0.93976507, 0.94191008, 0.94411319]),
#  'test_mae': array([0.73496285, 0.73180698, 0.7217913 , 0.72560867, 0.72723372])
#  
# KNNBasic - {'test_rmse': array([0.96725363, 0.97125242, 0.97014741, 0.96873569, 0.96112459]),
#  'test_mae': array([0.74064758, 0.74791949, 0.74719143, 0.74401927, 0.7398144 ])
#  
# KNNWithMeans - {'test_rmse': array([0.91904553, 0.91733627, 0.92334314, 0.92268939, 0.91496662]),
#  'test_mae': array([0.70309962, 0.70265884, 0.70387752, 0.70920784, 0.70105248])
#  
# KNNWithZScore - {'test_rmse': array([0.93017316, 0.92175447, 0.90740569, 0.91702126, 0.91801755]),
#  'test_mae': array([0.70783427, 0.70005986, 0.69185356, 0.69883193, 0.69449284])
#  
# KNNBaseline - {'test_rmse': array([0.89283459, 0.89803848, 0.88978505, 0.89786181, 0.90196116]),
#  'test_mae': array([0.68536267, 0.68852963, 0.6818973 , 0.68884985, 0.68979706])
#  
# and chose KNNBaseline (SVD algorithm shows similar results) as it has the best performance (RMSE - 0.89) and (MAE - 0.69)

# In[ ]:


alg = KNNBaseline()
cross_validate(alg, data, measures=['RMSE', 'MAE'])


# In[ ]:


trainset = data.build_full_trainset()
alg.fit(trainset)


# In[ ]:


alg.predict(1, 39)


# The 'getForecast' function gives a predicted rating that will drive a movie to specific users based on the ratings it has given to other movies and based on the ratings that those films have similarly rated.

# In[ ]:


def getForecast(userId, title):
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[1:11]
    
    show(title)
    
    movie_indices = [i[0] for i in sim_scores]
    
    movies = mainList.iloc[movie_indices][['id', 'title']]
    movies['id'] = movies['id'].astype(int)
    
    def getEst(item):
        return alg.predict(userId, item['id']).est
    
    movies['est'] = movies.apply(getEst, axis=1)
    
    return movies.head(10)


# In[ ]:


getForecast(5,'From Dusk Till Dawn')


# In[ ]:


getForecast(78,'From Dusk Till Dawn')


# In[ ]:


getForecast(76,'Twelve Monkeys')


# In[ ]:


getForecast(32,'Twelve Monkeys')


# # **K-Means Clustering**

# Read data

# In[ ]:


df = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv', low_memory=False)


# Only keep the numeric columns for our analysis

# In[ ]:


df.drop(df.index[19730],inplace=True)
df.drop(df.index[29502],inplace=True)
df.drop(df.index[35585],inplace=True)

df_numeric = df[['budget','popularity','revenue','runtime','vote_average','vote_count','title']]


# In[ ]:


df_numeric.head()


# Drop all the rows with null values

# In[ ]:


df_numeric.dropna(inplace=True)

df_numeric.isnull().sum()


# Take only the movies that have more than 30 votes.

# In[ ]:


df_numeric = df_numeric[df_numeric['vote_count']>30]


# Normalize the data with MinMax scaling provided by sklearn

# In[ ]:


minmax_processed = preprocessing.MinMaxScaler().fit_transform(df_numeric.drop('title',axis=1))
df_numeric_scaled = pd.DataFrame(minmax_processed, index=df_numeric.index, columns=df_numeric.columns[:-1])


# Find optimal k.
# We are dealing with tradeoff between cluster size(hence the computation required) and the relative accuracy

# In[ ]:


Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]

score = [kmeans[i].fit(df_numeric_scaled).score(df_numeric_scaled) for i in range(len(kmeans))]

pl.plot(Nc,score)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()


# In[ ]:


kmeans = KMeans(n_clusters=5)
kmeans.fit(df_numeric_scaled)
df_numeric['cluster'] = kmeans.labels_

df_numeric.head(20)


# Let's see cluster sizes.

# In[ ]:


plt.figure(figsize=(12,7))
axis = sn.barplot(x=np.arange(0,5,1),y=df_numeric.groupby(['cluster']).count()['budget'].values)
x=axis.set_xlabel("Cluster Number")
x=axis.set_ylabel("Number of movies")


# Let's look at the cluster statistics.

# In[ ]:


df_numeric.groupby(['cluster']).mean()


# # **XGBoost**

# In[ ]:


df = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv', low_memory=False)

df_numeric = df[['revenue','runtime','vote_average','vote_count']]
df_numeric.dropna(inplace=True)
x_data = df_numeric[['revenue','runtime','vote_count']]
y_data = df_numeric['vote_average']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

D_train = xgb.DMatrix(X_train, label=y_train)
D_test = xgb.DMatrix(X_test, label=y_test)

param = { 
    "silent":True,"eta":0.01,'subsample': 0.75,'colsample_bytree': 0.7,"max_depth":7, 'metric': 'rmse'} 

steps = 20 
model = xgb.train(param, D_train, steps)

preds = model.predict(D_test)


# # **Linear Regression**

# In[ ]:


df = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv', low_memory=False)
df = df.dropna()
df = df[['vote_average','revenue']]
df = df.loc[df['vote_average'] > 5]
df = df.loc[df['revenue'] < 1500000000]
df = df.loc[df['revenue'] > 1000000]

sns.lmplot(x ="vote_average", y ="revenue", data = df, order = 2, ci = None) 


# In[ ]:


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 20, test_size = 0.5)

model = LinearRegression()
model.fit(X_train, y_train)

model.score(X_test, y_test)
pred1 = model.predict(X_test) 
plt.scatter(X_test, y_test, color ='b') 
plt.plot(X_test, pred1, color ='k') 
  
plt.show() 


# In[ ]:


pred1 = model.predict(X_test) 
plt.scatter(X_test, y_test, color ='b') 
plt.plot(X_test, pred1, color ='k') 
  
plt.show() 


# # **Time series**

# In[ ]:


df = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv', low_memory=False)
df = df[['release_date', 'revenue']];
df = df.loc[df['revenue'] > 1000000]
df = df.loc[df['revenue'] < 1500000000]
df = df.loc[df['release_date'] > "2012-01-21"]
df = df.sort_values(by='release_date',ascending=True)
df.plot(figsize=(12,6), x='release_date',y='revenue')
df.reset_index(drop=True)
df.head(5)


# In[ ]:


split_date = '2015-01-01'
train_data = df[df['release_date'] <= split_date].copy()
test_data = df[df['release_date'] > split_date].copy()


# In[ ]:


model = Prophet()
prophetData = train_data.reset_index().rename(columns={'release_date':'ds','revenue':'y'})
model.fit(prophetData)

prophetTestData = test_data.reset_index().rename(columns={'release_date':'ds','revenue':'y'})
forecast = model.predict(df=prophetTestData)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()


# In[ ]:


fig1 = model.plot(forecast)


# In[ ]:


fig2 = model.plot_components(forecast)

