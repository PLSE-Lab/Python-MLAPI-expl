#!/usr/bin/env python
# coding: utf-8

# # Movie Recommendation System & Rating Predictor

# In this notebook, I will attempt at implementing a few recommendation algorithms (content based, popularity based and collaborative filtering) and try to build an ensemble of these models to come up with our final recommendation system. With us, we have two MovieLens datasets.
# 
# * **The Full Dataset:** Consists of 26,000,000 ratings and 750,000 tag applications applied to 45,000 movies by 270,000 users. Includes tag genome data with 12 million relevance scores across 1,100 tags.
# * **The Small Dataset:** Comprises of 100,000 ratings and 1,300 tag applications applied to 9,000 movies by 700 users.
# 
# I will build a Simple Recommender using movies from the *Full Dataset* whereas all personalised recommender systems will make use of the small dataset (due to the computing power I possess being very limited). As a first step, I will build my simple recommender system.
# 
# **NB:** This is the modified version of Rounak Banik's implementation.
# 
# ---
# This notebook is organized as follows:
# 
# **1. Exploration**
# - 1.1 Keywords
# - 1.2 Filling factor: missing values
# 
# **2. Data Cleaning**
# - 2.1 Drop missing valued rows
#     
# **3. Recommendation Engine**
# - 3.1 Simple Recommender
#  - 3.1.1 Weight sorting
#  - 3.1.2 Top movies
# - 3.2 Content Based Recommender
#  - 3.2.1 Movie metadata based filtering
# - 3.3 Collaborative Filtering
# 
# **4. Conclusion: possible improvements and points to adress**

# ## Exploration
# ---
# Let's import necessary modules first

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD
from sklearn import svm
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import datetime
import string


import warnings; warnings.simplefilter('ignore')
THRESHOLD_PREDICTION = 1


# Some utility functions

# In[ ]:


def clean_sentence(s, concat=None):
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = s.lower()
    if concat:
        s = concat.join(s.split())
    return s

def _make_in_format(df):
    y = np.array(df['rating'])
    temp_x = df.drop('rating', axis=1)  
#     print(temp_x)
    #min-max normalization
#     temp_x = (temp_x-temp_x.mean())/(temp_x.max()-temp_x.min())
    X = np.array(temp_x)

    return X,y

def accuracy_score(y_test,predictions):
        correct = []
        for i in range(len(y_test)):
            if predictions[i]>=y_test[i]-THRESHOLD_PREDICTION and predictions[i]<=y_test[i]+THRESHOLD_PREDICTION:
                correct.append(1)
            else:
                correct.append(0)

        accuracy = sum(map(int,correct))*1.0/len(correct)
        return accuracy


# ### 1.1 Keywords
# 
# We will first load the movies metadata and try to explore it.

# In[ ]:


md = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv')
md.head()


# We can see that, there are 24 columns in each row. But in this notebook, I will only consider the following columns in my calculation-
# 
# - Overview
# - Cast-crew
# - Director
# - Genre
# - Vote count
# - Release date
# - Revenue
# - Vote average
# 
# Now load the **credits**,**keywords** and **ratings** dataset-

# In[ ]:


credits = pd.read_csv('../input/the-movies-dataset/credits.csv')
keywords = pd.read_csv('../input/the-movies-dataset/keywords.csv')
ratings = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')


# In[ ]:


credits.head(5)


# In[ ]:


keywords.head(5)


# In[ ]:


ratings.head(5)


# We will now take a look at the frequency of keywords. This will help us to get the popular keywords.

# In[ ]:


keywords_len = len(keywords)
keywords_dict = {}
keywords_id_dict = {}

for it in range(keywords_len):
    keywords_arr = keywords.iloc[it]['keywords']
    keywords_arr = eval(keywords_arr)
    keywords_id_dict[keywords.iloc[it]['id']]=""
    for iit in range(len(keywords_arr)):
        keywords_id_dict[keywords.iloc[it]['id']] = keywords_id_dict[keywords.iloc[it]['id']] + clean_sentence(keywords_arr[iit]['name']) + " "
        if keywords_dict.get(keywords_arr[iit]['name']):
            keywords_dict[keywords_arr[iit]['name']] = keywords_dict[keywords_arr[iit]['name']]+1
        else:
            keywords_dict[keywords_arr[iit]['name']]=1


# In[ ]:


# sort in ascending order of occurence
keyword_occurences = []
for k,v in keywords_dict.items():
    keyword_occurences.append([k,v])
keyword_occurences.sort(key = lambda x:x[1], reverse = True)


# In[ ]:


# HISTOGRAMS
fig = plt.figure(1, figsize=(18,13))
ax = fig.add_subplot(1,1,1)
trunc_occurences = keyword_occurences[0:50]
y_axis = [i[1] for i in trunc_occurences]
x_axis = [k for k,i in enumerate(trunc_occurences)]
x_label = [i[0] for i in trunc_occurences]
plt.xticks(rotation=85, fontsize = 15)
plt.yticks(fontsize = 15)
plt.xticks(x_axis, x_label)
plt.ylabel("Nb. of occurences", fontsize = 18, labelpad = 10)
ax.bar(x_axis, y_axis, align = 'center', color='g')
#_______________________
plt.title("Keywords popularity",bbox={'facecolor':'k', 'pad':5},color='w',fontsize = 25)
plt.show()


# ### 1.2 Filling factor: missing values
# 
# The dataset consists in 45466 films or TV series which are described by 24 variables. As in every analysis, at some point, we will have to deal with the missing values and as a first step, I determine the amount of data which is missing in every variable:

# In[ ]:


def missing_factor(p_df):
    missing_df = p_df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df['filling_factor'] = (p_df.shape[0] 
                                - missing_df['missing_count']) / p_df.shape[0] * 100
    
    return missing_df


# In[ ]:


meta_missing = missing_factor(md)
meta_missing.sort_values('filling_factor').reset_index(drop = True)


# In[ ]:


keywords_missing = missing_factor(keywords)
keywords_missing.sort_values('filling_factor').reset_index(drop = True)


# In[ ]:


credits_missing = missing_factor(credits)
credits_missing.sort_values('filling_factor').reset_index(drop = True)


# In[ ]:


ratings_missing = missing_factor(ratings)
ratings_missing.sort_values('filling_factor').reset_index(drop = True)


# 
# 
# We can see that most of the variables are well filled since only 3 of them have a filling factor below 97%. Later we will see that those 3 columns are not used in any calculation.
# 
# 
# 
# 

# ## Data Cleaning
# 
# We've saw that the main data columns in the dataset is well filled. So, we've decided to drop the row which are not well filled in the concerned column.
# 
# ### 2.1 Drop missing valued rows
# 
# Only the metadata set have some missing valued rows in the following columns of concern:
# 
# - Overview
# - Vote count
# - Release date
# - Revenue
# - Title
# - Vote average
# 
# So, we will drop the rows which have missing values in the specified columns

# In[ ]:


md.replace(r'^\s*$', np.NaN, regex=True)
md = md.dropna(subset=['overview'])
md = md.dropna(subset=['vote_count'])
md = md.dropna(subset=['release_date'])
md = md.dropna(subset=['revenue'])
md = md.dropna(subset=['title'])
md = md.dropna(subset=['vote_average'])


# In[ ]:


md.shape


# Now, we will create a movie dictionary with movie id as key and another dictionary with important feature as value

# In[ ]:


movie_id_dict = {}
movies_data_len = md.shape[0]
train_dataset = pd.DataFrame()
avg_popularity = 0
avg_vote_count = 0
avg_vote_average = 0
avg_revenue = 0

totally_filled_data_count = 1

popularity_total = 0
vote_count_total = 0
vote_average_total = 0
revenue_total = 0

flag = False

for it in range(movies_data_len):
    if md.iloc[it]['popularity'] and isinstance(md.iloc[it]['popularity'], float) and np.isnan(md.iloc[it]['popularity'])==False:
        if md.iloc[it]['vote_count'] and isinstance(md.iloc[it]['vote_count'], float) and np.isnan(md.iloc[it]['vote_count'])==False:
            if md.iloc[it]['vote_average']  and isinstance(md.iloc[it]['vote_average'], float) and np.isnan(md.iloc[it]['vote_average'])==False:
                if md.iloc[it]['revenue'] and isinstance(md.iloc[it]['revenue'], float) and np.isnan(md.iloc[it]['revenue'])==False:
                    popularity_total += md.iloc[it]['popularity']
                    vote_count_total += md.iloc[it]['vote_count']
                    vote_average_total += md.iloc[it]['vote_average']
                    revenue_total += md.iloc[it]['revenue']
                    totally_filled_data_count += 1
                    
    cur_genres = eval(md.iloc[it]['genres'])
    concated_genres = "" 
    for git in range(len(cur_genres)):
        item = cur_genres[git]['name']
        concated_genres += clean_sentence(item, "_")+' '
        
    
    movie_id_dict[int(md.iloc[it]['id'])] = {'popularity': md.iloc[it]['popularity'], 
                                        'vote_count': md.iloc[it]['vote_count'], 
                                        'vote_average': md.iloc[it]['vote_average'],
                                        'revenue': md.iloc[it]['revenue'],
                                        'genres': concated_genres,
                                        'overview': clean_sentence(md.iloc[it]['overview']),
                                        'title': clean_sentence(md.iloc[it]['title'])}
    
avg_popularity = popularity_total/totally_filled_data_count
avg_vote_count = vote_count_total/totally_filled_data_count
avg_vote_average = vote_average_total/totally_filled_data_count
avg_revenue = revenue_total/totally_filled_data_count


# In[ ]:


movie_id_dict[15602]


# ## Recommendation Engine
# ---
# Now we will build some recommendation engine.

# ### 3.1 Simple Recommender
# 
# The Simple Recommender offers generalized recommnendations to every user based on movie popularity and (sometimes) genre. The basic idea behind this recommender is that movies that are more popular and more critically acclaimed will have a higher probability of being liked by the average audience. This model does not give personalized recommendations based on the user.

# #### 3.1.1 Weight sorting
# 
# The implementation of this model is extremely trivial. All we have to do is sort our movies based on ratings and popularity and display the top movies of our list. As an added step, we can pass in a genre argument to get the top movies of a particular genre. 

# In[ ]:


md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# We will use IMDB's *weighted rating* formula to construct my chart. Mathematically, it is represented as follows:
# 
# Weighted Rating (WR) = $(\frac{v}{v + m} . R) + (\frac{m}{v + m} . C)$
# 
# where,
# * *v* is the number of votes for the movie
# * *m* is the minimum votes required to be listed in the chart
# * *R* is the average rating of the movie
# * *C* is the mean vote across the whole report
# 
# The next step is to determine an appropriate value for *m*, the minimum votes required to be listed in the chart. We will use **90th percentile** as our cutoff. In other words, for a movie to feature in the charts, it must have more votes than at least 90% of the movies in the list.
# 
# We will build our overall Top 250 Chart and will define a function to build charts for a particular genre. Let's begin!

# In[ ]:


vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
C


# In[ ]:


m = vote_counts.quantile(0.90)
m 


# In[ ]:


md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


# In[ ]:


qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
qualified.shape


# Therefore, to qualify to be considered for the chart, a movie has to have at least **434 votes**. We also see that the average rating for a movie on is **5.244** on a scale of 10. **4462** Movies qualify to be on our chart.

# In[ ]:


def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[ ]:


qualified['wr'] = qualified.apply(weighted_rating, axis=1)


# In[ ]:


qualified = qualified.sort_values('wr', ascending=False).head(250)


# #### 3.1.2 Top movies

# In[ ]:


qualified.head(10)


# Let us now construct our function that builds charts for particular genres. For this, we will use relax our default conditions to the **85th** percentile instead of 90. 

# In[ ]:


s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = md.drop('genres', axis=1).join(s)


# In[ ]:


def build_chart(genre, percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    
    return qualified


# Let us see our method in action by displaying the Top **15 Romance Movies** (Romance almost didn't feature at all in our Generic Top Chart despite  being one of the most popular movie genres).

# In[ ]:


build_chart('Romance').head(15)


# The top romance movie according to our metrics is Bollywood's **Dilwale Dulhania Le Jayenge**.

# ### 3.2 Content Based Recommender
# 
# To personalise our recommendations more, we are going to build an engine that computes similarity between movies based on certain metrics and suggests movies that are most similar to a particular movie that a user liked. Since we will be using movie metadata (or content) to build this engine, this also known as **Content Based Filtering.** We will use the TMDB movies for this purpose because of computational complexity.
# 

# In[ ]:


links_small = pd.read_csv('../input/the-movies-dataset/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
# md.shape


# In[ ]:


md['id'] = md['id'].astype('int')


# In[ ]:


smd = md[md['id'].isin(links_small)]
smd.shape


# In[ ]:


#Check EDA Notebook for how and why I got these indices.
smd['id'] = smd['id'].astype('int')


# We have **9087** movies avaiable in our small movies metadata dataset which is 5 times smaller than our original dataset of movies.

# #### 3.2.1 Movie metadata based filtering
# 
# Let us first try to build a recommender using movie metadata(overview, ). We do not have a quantitative metric to judge our machine's performance so this will have to be done qualitatively.

# In[ ]:


credits_id_dict = {}

for cit in range(len(credits)):
    cast_item = eval(credits.iloc[cit]['cast'])
    crew_item = eval(credits.iloc[cit]['crew'])
    concat_cast_crew = ""
    
    for cast_it in range(len(cast_item)):
        concat_cast_crew += clean_sentence(cast_item[cast_it]['name'], "_")+" "
        
    for crew_it in range(len(crew_item)):
        concat_cast_crew += clean_sentence(crew_item[crew_it]['name'], "_")+" "
        
    credits_id_dict[credits.iloc[cit]['id']] = concat_cast_crew


# In[ ]:


smd['description'] = smd['overview']

for mvit in range(len(smd)):
    mid = smd.iloc[mvit]['id']
    smd.iloc[mvit]['description'] = ""
    
    if movie_id_dict.get(mid):
        smd.iloc[mvit]['description'] += movie_id_dict[mid]['genres'] + " " + movie_id_dict[mid]['overview']
        
    if keywords_id_dict.get(mid):
        smd.iloc[mvit]['description'] += keywords_id_dict[mid] + " "
        
    if credits_id_dict.get(mid):
        smd.iloc[mvit]['description'] += credits_id_dict[mid] + " "


# In[ ]:


smd.shape


# In[ ]:


tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])


# In[ ]:


# tfidf_matrix.shape
tfidf_matrix


# #### Cosine Similarity
# 
# We will be using the Cosine Similarity to calculate a numeric quantity that denotes the similarity between two movies. Mathematically, it is defined as follows:
# 
# $cosine(x,y) = \frac{x. y^\intercal}{||x||.||y||} $
# 
# Since we have used the TF-IDF Vectorizer, calculating the Dot Product will directly give us the Cosine Similarity Score. Therefore, we will use sklearn's **linear_kernel** instead of cosine_similarities since it is much faster.

# In[ ]:


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[ ]:


cosine_sim.shape


# We now have a pairwise cosine similarity matrix for all the movies in our dataset. The next step is to write a function that returns the 30 most similar movies based on the cosine similarity score.

# In[ ]:


smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])


# In[ ]:


def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# We're all set. Let us now try and get the top recommendations for a few movies and see how good the recommendations are.

# In[ ]:


get_recommendations('The Godfather').head(20)


# In[ ]:


get_recommendations('The Dark Knight').head(10)


# We see that for **The Dark Knight**, our system is able to identify it as a Batman film and subsequently recommend other Batman films as its top recommendations. But unfortunately, that is all this system can do at the moment.

# ### Collaborative Filtering
# 
# Our content based engine suffers from some severe limitations. It is only capable of suggesting movies which are *close* to a certain movie. That is, it is not capable of capturing tastes and providing recommendations across genres.
# 
# Also, the engine that we built is not really personal in that it doesn't capture the personal tastes and biases of a user. Anyone querying our engine for recommendations based on a movie will receive the same recommendations for that movie, regardless of who s/he is.
# 
# Therefore, in this section, we will use a technique called **Collaborative Filtering** to make recommendations to Movie Watchers. Collaborative Filtering is based on the idea that users similar to a me can be used to predict how much I will like a particular product or service those users have used/experienced but I have not.
# 
# We will not be implementing Collaborative Filtering from scratch. Instead, we will use the **sklearn's svr**.

# In[ ]:


reader = Reader()


# In[ ]:


from numpy import nan


# In[ ]:


rating_data_len = ratings.shape[0]

modified_trainset = pd.DataFrame(index=range(rating_data_len), 
                                columns=['userId', 'movieId','popularity',
                                        'vote_count','vote_average','revenue', 
                                        'rating'])
popularityArr = [0]*rating_data_len
voteCountArr = [0]*rating_data_len
voteAverageArr = [0]*rating_data_len
revenueArr = [0]*rating_data_len


for it in range(rating_data_len):
    movie_id = int(ratings.iloc[it]['movieId'])
    movie_metadata = movie_id_dict.get(movie_id)
    
    if movie_metadata:
        temp = movie_metadata['popularity']
        if isinstance(temp, str):
            temp = float(temp)
            
        if np.isnan(temp):
            popularityArr[it] = avg_popularity
        else:
            popularityArr[it] = temp

            
            
        temp = movie_metadata['vote_count']
        if isinstance(temp, str):
            temp = float(temp)
            
        if np.isnan(temp):
            voteCountArr[it] =  avg_vote_count
        else:
            voteCountArr[it] = temp

            
            
        temp = movie_metadata['vote_average']
        if isinstance(temp, str):
            temp = float(temp)
            
        if np.isnan(temp):
            voteAverageArr[it] = avg_vote_average
        else:
            voteAverageArr[it] = temp

        
        
        temp = movie_metadata['revenue']
        if isinstance(temp, str):
            temp = float(temp)
            
        if np.isnan(temp):
            revenueArr[it] =  avg_revenue
        else:
            revenueArr[it] = temp
    else:
        popularityArr[it] = avg_popularity
        voteCountArr[it] =  avg_vote_count
        voteAverageArr[it] = avg_vote_average
        revenueArr[it] =  avg_revenue


# In[ ]:


modified_trainset['userId'] = ratings['userId']*100
modified_trainset['movieId'] = ratings['movieId']
modified_trainset['popularity'] = popularityArr
modified_trainset['vote_count'] = voteCountArr
modified_trainset['vote_average'] = voteAverageArr
modified_trainset['revenue'] = revenueArr
modified_trainset['rating'] = ratings['rating']


# In[ ]:


modified_trainset.head(12)


# In[ ]:


X,y = _make_in_format(modified_trainset)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=1)


# **SVR**

# In[ ]:


svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_system = svr_rbf.fit(X_train, y_train)


# In[ ]:


y_pred_svr = svr_system.predict(X_test)
accuracy_score(y_test, y_pred_svr)


# In[ ]:


def make_prediction_svr(userId, movieId):
    x_test = modified_trainset.loc[modified_trainset['movieId'] == movieId].head(1)
    if len(x_test)==0:
        print('No movie found')
    else:
        y_prediction = svr_system.predict([[userId, movieId, x_test['popularity'], x_test['vote_count'], x_test['vote_average'], x_test['revenue']]])
        print(y_prediction)


# In[ ]:


make_prediction_svr(100, 1029)


# ## Conclusion
# 
# In this notebook, I have built 4 different recommendation engines based on different ideas and algorithms. They are as follows:
# 
# 1. **Simple Recommender:** This system used overall TMDB Vote Count and Vote Averages to build Top Movies Charts, in general and for a specific genre. The IMDB Weighted Rating System was used to calculate ratings on which the sorting was finally performed.
# 2. **Content Based Recommender:** We built a content based engine; that took movie overview, cast, crew, genre and keywords to come up with predictions.
# 3. **Collaborative Filtering:** We used the powerful sklearn library to build a collaborative filter based on support vector machine. The engine gave estimated ratings for a given user and movie.

# In[ ]:




