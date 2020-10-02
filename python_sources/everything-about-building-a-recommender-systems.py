#!/usr/bin/env python
# coding: utf-8

# <h2>**RECOMMENDER SYSTEMS IN OUR DAY TO DAY LIFE:**</h2>
#  
#  
#  
#  <h3>Introduction</h3>
#  
#  The main objective of this kernel is make sure even a naive person without having much of ml knowledge can understand what was going and after completing this notebook i am confident that you can build your own recommender models using ml.Be confident and don't bother by seeing the size of this notebook it contains pure simple language so everyone can understand it......I explained each and every step of  model building... So please spend time and try it complete it ....If there are any suggestions you can mention it in comments and in the middle of the notebook you will some links please checkout those links also......
# 
# If you find this notebook useful please upvote it.
# 
# 
#  This kernel will give all the insights about building Recommender systems from scratch in case you have no knowledge about recommender sytsems you can understand fully and indepth concepts of Recommender systems..
#         
#  In our day to day life from reading an article in mobile after waking up to watching some
#                   movies before sleep the one thing that plays a crucial role is recommender systems.
#                   Recommender Systems is used to recommend the shopping items,news articles,movies,songs..etc..,.
#                   So the Demand for this Recommender Systems is increasing a lot day by day.
#                   Now let's dive into the learning process.After learning this kernal, i am sure that you are going to build some awesome
#                   recommender systems.
#                   And i try to make the concepts simple and understandable to some extent that is possible for me.
#            

# The rapid growth of data collection has led to a new era of information. Data is being used to create more efficient 
#                  systems and this is where Recommendation Systems comes into play. 
#                  Recommendation Systems are a type of information filtering systems as they improve the quality of search results
#                  and provides items that are more relevant to the search item or related to the search history of the user.
#                              They are used to predict the rating or preference that a user would give to an item.Companies like Netflix 
#                  and Spotify depend highly on the effectiveness of their recommendation engines for their business and sucees.
#                  
#   If find this kernel helpful please upvote it....
#                            

#    In this kernel we'll be building a baseline Movie Recommendation System using TMDB 5000 Movie Dataset.
#             
#    **Lets start**
#            There are mainly three kinds of recommender systems:-
#            
#   **1)<h1>Demographic Filtering</h1>**- They offer generalized recommendations to every user, based on movie popularity and/or genre. The System recommends the same movies to users with similar demographic features. Since each user is different , this approach is considered to be too simple. The basic idea behind this system is that movies that are more popular and critically acclaimed will have a higher probability of being liked by the average audience.
#            When we open some app in our phone it asks to access our location.Based on our location and people around it will recommend the things which we may like and this is basic recommender present in any application.
#            
#   **2)<h1>Content Based Filtering</h1>**- They suggest similar items based on a particular item. This system uses item metadata, such as genre, director, description, actors, etc. for movies, to make these recommendations. The general idea behind these recommender systems is that if a person liked a particular item, he or she will also like an item that is similar to it.
#             This can be seen in applications like Netflix,Facebook Watch etc..,. which recommend us the next movie or video based on the director,hero etc..,.
#             
#   **3)<h1>Collaborative Filtering</h1>**- This system matches persons with similar interests and provides recommendations based on this matching. Collaborative filters do not require item metadata like its content-based counterparts.
#             This is the most sophisticated personalized recommendation that means it takes into account what user likes and not likes.
#             The main example for this is Google Ads.
# 

# 1. First let's import the datasets that are needed:-

# In[ ]:


import pandas as pd
import numpy as np
df1=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')
df2=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')


# * Now examine the dataframes:

# In[ ]:


df1.shape


# In[ ]:


df2.shape


# 1.    The first dataset contains the following features:-
#                     movie_id - A unique identifier for each movie.
#                     cast - The name of lead and supporting actors.
#                     crew - The name of Director, Editor, Composer, Writer etc.

#     The second dataset has the following features:-
# 
#     budget - The budget in which the movie was made.
#     genre - The genre of the movie, Action, Comedy ,Thriller etc.
#     homepage - A link to the homepage of the movie.
#     id - This is infact the movie_id as in the first dataset.
#     keywords - The keywords or tags related to the movie.
#     original_language - The language in which the movie was made.
#     original_title - The title of the movie before translation or adaptation.
#     overview - A brief description of the movie.
#     popularity - A numeric quantity specifying the movie popularity.
#     production_companies - The production house of the movie.
#     production_countries - The country in which it was produced.
#     release_date - The date on which it was released.
#     revenue - The worldwide revenue generated by the movie.
#     runtime - The running time of the movie in minutes.
#     status - "Released" or "Rumored".
#     tagline - Movie's tagline.
#     title - Title of the movie.
#     vote_average - average ratings the movie recieved.
#     vote_count - the count of votes recieved.

# As we see in the first dataframe we have movie_id and in second we have id.
# so to work on this dataframes in combination we need to rename that movie_id to id.
# And now we will do it as:

# In[ ]:


df1.columns = ['id','tittle','cast','crew']


# In[ ]:


df1.head()


#   yup....we got id in df1 now we can start working.
#         Now to work in this data we need to combine this data.And we can do it very easily with the help of the merge()
#         function and this merging is done on 'id' column.As it is common in both the data sets.So now lets do it
#         
# *   2.Merging dataframes on 'id' column

# In[ ]:


df2=df2.merge(df1,on="id")
df2.head()


# Wow .... Now we done with second step that is we finished the merging of both the dataFrames and call head() to have some idea on dataset.Let's start building recommender systems:-
#     

# **Demographic Filtering -**
#         
# 1.   Before getting started with this :
#         
#         *1.we need a metric to score or rate movie(criteria based on which we can define which are best movies)*
#         
#         *2.Calculate the score for every movie*
#         
#         *3.Sort the scores and recommend the best rated movie to the users.*
#             We can use the average ratings of the movie as the score but using this won't be fair enough since a movie with 8.9 average
#             rating and only 3 votes cannot be considered better than the movie with 7.8 as as average rating but 40 votes. So, I'll be 
#             using IMDB's weighted rating (wr) which is given as :-
#                                                      Weighted Average(WR):-((v/v+m)*R)+((m/m+v)*C)
#                            where
#                            v- number of votes for the movie
#                            m-minimum votes required to be listed in the chart
#                            R-average rating of the movie
#                            C-mean vote across the whole report
#          
#         we have v(vote_count) and R(average rating of movie).Now calculate C
#       

# In[ ]:


c=df2['vote_average'].mean()
print(c)


#   Now Filter out only the best movies that means Finding m:
#             which means we need to set some benchmark and the movies that are above it will be taken into consideration.
#             For doing this we are using the quantile function to get the top movies..... We will use 90th percentile as our cutoff. In 
#             other words, for a movie to feature in the charts, it must have more votes than at least 90% of the movies in the list.
#             

# In[ ]:


m=df2['vote_count'].quantile(0.9)
m


#    Here 'm' specifies that for a movie to be in our final list it must have more than 'm' votes:

# In[ ]:


qualified_movies=df2.copy().loc[df2['vote_count']>=m]
qualified_movies.shape


#    So 481 movies have been through into our final pool and now we will give score to each of this movies
#              To do this, we will define a function, imdbscore() and define a new feature score, of which we'll calculate the value 
#              by applying this function to our DataFrame of qualified movies:
#              Let's do this:

# In[ ]:


def imdbscore(x,m=m,c=c):
    v=x['vote_count']
    r=x['vote_average']
    #now remember the imdb formula 
    return (v/(v+m) * r) + (m/(m+v) * c)


# In[ ]:


#now apply this to our dataframe qualified_movies through apply() function.
qualified_movies['score']=qualified_movies.apply(imdbscore,axis=1)


#             Finally let's now sort the movies scores so that we can get the top movies.As this is dataframe,
#             we use sorted_values() function to sort then print the top movies using the head.
#             

# In[ ]:


#Sort movies based on score calculated above
qualified_movies = qualified_movies.sort_values('score', ascending=False)

#Print the top 15 movies
qualified_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)


#    Wow guys ....... we had done it and this is the basic recommender system used in any application at the time 
#                 you installed the app. As it doesn't have any knowledge about you it shows this generic type recommendation.
#                 Now it's time to have some fun...
#                 Let's checkout the popular movies:-

# In[ ]:


popular=df2.sort_values(by='popularity',ascending=False)
popular[['title', 'vote_count', 'vote_average']].head()


#    So this are the movies that your going to see in popular movies category or we can find them in the trending 
#             tag.But...hold on guys
#             these demographic recommender provide a general chart of recommended movies to all the users. They are not sensitive to the 
#             interests and tastes of a particular user. This is when we move on to a more refined system- Content Basesd Filtering.

# **Content Based Filtering**:-
#         
#    In this recommender system the content of the movie (overview, cast, crew, keyword, tagline etc) is used to find its similarity with other movies. Then the movies that are most likely to be similar are recommended.
#    That means it will be like if we watched Harrypotter part1 then we will tend to watch remaining parts of it as well.That is what
#    this content based filtering will do.Now we will construct it step by step:-
#    
#    First we are going to build an Plot description based Recommender than we will go for credit,crew,genre based Recommender...
#    
#    Let's start ...
#    
#         

# **Plot description based Recommender**
# 
# We will compute pairwise similarity scores for all movies based on their plot descriptions and recommend movies based on that similarity score. The plot description is given in the overview feature of our dataset. Let's take a look at the overviews given..

# In[ ]:


df2['overview'].head(5)


#   Now we'll compute **Term Frequency-Inverse Document Frequency (TF-IDF)** vectors for each overview.
#   
#   Now if you are wondering what is term frequency , *it is the relative frequency of a word in a document and is given as (term instances/total instances). Inverse Document Frequency is the relative count of documents containing the term is given as log(number of documents/documents with term) The overall importance of each word to the documents in which they appear is equal to TF * IDF.*
#   
#  In simple terms:-
#  term frequency:-number of times word occuring in document to total words in document.
#  
#  This,number of times that is the word count is calculated using the **countvectorizer()**.Then using the Inverse Document Frequency it will normalize the count.
#  
#  This will give you a matrix where each column represents a word in the overview vocabulary (all the words that appear in at least one document) and each row represents a movie, as before.The IDF talked above is done to reduce the importance of words that occur frequently in plot overviews and therefore,reduce their significance in computing the final similarity score.
#  
#  Now if you are wandering how to do this....calm down guys,the scikit-learn makes our life simple:scikit-learn gives you a built-in TfIdfVectorizer class that produces the TF-IDF matrix in a couple of lines.
#  
#   

# In[ ]:


#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'.This is very important because they occurs in all the
#overviews and that will result in getting false similarity scores
tfidf = TfidfVectorizer(stop_words='english')


#    check any null values are there:-

# In[ ]:


df2['overview'].isna().sum()


# Yes null values are present ... so,now try to replace them with empty string so our model won't get any errors

# In[ ]:


#Replace NaN with an empty string
df2['overview'] = df2['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2['overview'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape


# So what happened here is for every movie,the similarity scores with each and every word is found and are stored in a matrix.
# We see that over 20,000 different words were used to describe the 4800 movies in our dataset.
# 
# With this matrix in hand, we can now compute a similarity score. There are several candidates for this; such as the euclidean, the Pearson and the cosine similarity scores. There is no right answer to which score is the best. Different scores work well in different scenarios and it is often a good idea to experiment with different metrics.
# 
# We will be using the cosine similarity to calculate a numeric quantity that denotes the similarity between two movies. We use the cosine similarity score since it is independent of magnitude and is relatively easy and fast to calculate.
# 
# **Remember**This similarity is nothing but doing the dot product of the two movies that we are going to compare.In this we will take a movie and we will calculate the dot product(similarity score) with every other movie and we will give output the movie which resulted in getting the maximum similarity score with the movie to which we compared

#    Since we have used the TF-IDF vectorizer, calculating the dot product will directly give us the cosine similarity score(As i explained above). Therefore, we will use sklearn's linear_kernel() instead of cosine_similarities() since it is faster in calculating the dot product.

# In[ ]:


# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


#    We are going to define a function that takes in a movie title as an input and outputs a list of the 10 most similar movies. Firstly, for this, we need a reverse mapping of movie titles and DataFrame indices. In other words, we need a mechanism to identify the index of a movie in our metadata DataFrame, given its title.
#    
#    so we will use the series() constructor along with index parameter.This will help us in creating mapping between indices and the titles of the movies

# In[ ]:


#Construct a reverse map of indices and movie titles
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()
#if any duplicate indexes are present they will get eliminated 


# We are now in a good position to define our recommendation function. These are the following steps we'll follow :-
# 
# * Get the index of the movie given its title.
# * Get the list of cosine similarity scores for that particular movie with all movies. Convert it into a list of tuples where the first element is its index(index of movie) and the second is the similarity score.
# * Sort the aforementioned list of tuples based on the similarity scores; that is, the second element.
# * Get the top 10 elements of this list. Ignore the first element as it refers to self (the movie most similar to a particular movie is the movie itself).
# * Return the titles corresponding to the indices of the top elements.

# In[ ]:


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title(the movie for which we need recommendations)
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores.Here as it is list to sort --sorted() function is used.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df2['title'].iloc[movie_indices]


# We will now look at how it's going to work:-

# In[ ]:


get_recommendations('The Avengers')


#    So here one can observe that the movies which were recommeneded doesn't contain the other marvel or dc movies as they have 
#    different scripts .... It's obvious that different movies have different scripts but in general if a person watched Avengers: Age of
#    Ultron he tend to watch other marvel movies.So to overcome this thing we go for Credits, Genres and Keywords Based Recommender.

# **Credits, Genres and Keywords Based Recommender**
# 
#    It goes without saying that the quality of our recommender would be increased with the usage of better metadata(information about actors,film etc..,). That is exactly what we are going to do in this section. We are going to build a recommender based on the following metadata: the 3 top actors, the director, related genres and the movie plot keywords.
# 
# From the cast, crew and keywords features, we need to extract the three most important actors, the director and the keywords associated with that movie.
# 
# Right now, our data is present in the form of "stringified" lists , we need to convert it into a safe and usable structure

# In[ ]:


# Parse the stringified features into their corresponding python objects
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']#since we are considering actors,directors present in queue and genre and keywords about movie
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)


# Now start collecting the information that is about the director,actors and genre.

# In[ ]:


# Get the director's name from the crew feature. If director is not listed, return NaN
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# In[ ]:


# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []


# In[ ]:


# Define new director, cast, genres and keywords features that are in a suitable form.
df2['director'] = df2['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)


# In[ ]:


df2.head()


# So now we can see that the dataframe contains all the metadata required so we can now go to the next step:-
# 
# We need to do some preprocessing of data like convert the names and keyword instances into lowercase and strip all the spaces between them. This is done so that our vectorizer doesn't count the Johnny of "Johnny Depp" and "Johnny Galecki" as the same.

# In[ ]:


# Function to convert all strings to lower case and strip names of spaces
def clean_data(li):
    if isinstance(li, list):#To avoid errors
        return [str.lower(i.replace(" ", "")) for i in li]#for get_list() function.
    else:
        #Check if director exists. If not, return empty string
        if isinstance(li, str):
            return str.lower(li.replace(" ", ""))#for director
        else:
            return ''


# Now let's apply this to our data

# In[ ]:


# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    df2[feature] = df2[feature].apply(clean_data)


# Now we can build anykind of recommender system like recommender based on director or cast or genre etc..,. or we can combine them all
# and can build an recommender system which takes into account all the criteria

# **RECOMMENDER SYSTEM BASED ON METADATA** 
# 
# We are now in a position to create our "metadata solution", which is a string that contains all the metadata that we want to feed to our vectorizer (namely actors, director and keywords).
# 
# **NOTE**:-here we can prepare the solution with only genre or actors or keywords or combination of any two etc..,.Those are helpful when we are building recommender system based on certain creteria like based on genre or based on actors etc..,.But for now we are building the recommender system based on everything

# In[ ]:


def create_solution(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
df2['solution'] = df2.apply(create_solution, axis=1)


# The next steps are the same as what we did with our plot description based recommender. One important difference is that we use the CountVectorizer() instead of TF-IDF. This is because we do not want to down-weight the presence of an actor/director if he or she has acted or directed in relatively more movies.
# 
# That means we are not going normalize here and i think follow this link if you are not sure or not understood properly will helps you:-[difference between countvectorizer and tfidf](https://datascience.stackexchange.com/questions/25581/what-is-the-difference-between-countvectorizer-token-counts-and-tfidftransformer)
# 

# In[ ]:


# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')#not considering the words like 'the','a'etc..,.
count_matrix = count.fit_transform(df2['solution'])


# In[ ]:


# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)#we can use linear kernel as above said also


# In[ ]:


# Reset index of our main DataFrame and construct reverse mapping as before
df2 = df2.reset_index()#resetting
indices = pd.Series(df2.index, index=df2['title'])#now again mapping the movies and id's...


# In[ ]:


df2.head()


# You can see the indexing there .....Now we can reuse our get_recommendations() function by passing in the new cosine_sim2 matrix as your second argument.

# In[ ]:


get_recommendations('The Dark Knight Rises', cosine_sim2)


# So you can see that,Now our recommender system is working a lot better by taking into account all the metadata and if you want this same
# kind of recommender system for music you can create the solution by taking into account the singers,music director and genre..
# 
# And you like to add some more importance to hero or director or genre in the solution mixture join them more number of times so that there importance will increase

# Now let's start the final type and the most important and personalized recommender system
# 
# **Collaborative Filtering**
# 
# Our content based engine suffers from some severe limitations. It is only capable of suggesting movies which are close to a certain movie. That is, it is not capable of capturing tastes and providing recommendations across genres.
# 
# Also, the engine that we built is not really personal in that it doesn't capture the personal tastes and biases of a user. Anyone querying our engine for recommendations based on a movie will receive the same recommendations for that movie, regardless of who she/he is.
# 
# Therefore, in this section, we will use a technique called Collaborative Filtering to make recommendations to Movie Watchers. It is basically of two types:-
# 
# **User based filtering**- These systems recommend products to a user that similar users have liked. For measuring the similarity between two users we can either use pearson correlation or cosine similarity
# 
# **Item Based Collaborative Filtering** - Instead of measuring the similarity between users, the item-based CF recommends items based on their similarity with the items that the target user rated. Likewise, the similarity can be computed with Pearson Correlation or Cosine Similarity. The major difference is that, with item-based collaborative filtering, we fill in the blank vertically, as oppose to the horizontal manner that user-based CF does
# 
# Please once check out the following link to get better idea of gist i explained:-
# 
# [basics on User based and Item based filtering](https://hackernoon.com/introduction-to-recommender-system-part-1-collaborative-filtering-singular-value-decomposition-44c9659c5e75)
# 
# Please look at the tables explained in link for better understanding.
# 
# After checking out the link:-we can proceed further now....
# 
# Item based CF successfully avoids the problem posed by dynamic user preference as item-based CF is more static. However, several problems remain for this method. First, the main issue is scalability. The computation grows with both the customer and the product. The worst case complexity is **O**(mn) with m users and n items. In addition, sparsity is another concern.In extreme cases, we can have millions of users and the similarity between two fairly different movies could be very high simply because they have similar rank for the only user who ranked them both.
# 
# **SINGLE VALUED DECOMPOSITION**
# 
# One way to handle the scalability and sparsity issue created by CF is to leverage a **latent factor model** to capture the similarity between users and items. Essentially, we want to turn the recommendation problem into an optimization problem. We can view it as how good we are in predicting the rating for items given a user. One common metric is Root Mean Square Error (RMSE). The lower the RMSE, the better the performance.
# 
# Now talking about latent factor you might be wondering what is it ?It is a broad idea which describes a property or concept that a user or an item have. For instance, for music, latent factor can refer to the genre that the music belongs to. SVD decreases the dimension of the utility matrix by extracting its latent factors. Essentially, we map each user and each item into a latent space with dimension r. Therefore, it helps us better understand the relationship between users and items as they become directly comparable.
# 
# Here what it does is it coverts the matrix users vs products into two:-
# 
# It divides into two matrices in such a way that users will mapped into some r factors and the products is also mapped to same r factors
# so they are going to map to same
# 
# please check the following link of **MIT LECTURE** of just 11min you will get complete idea about it:-
# 
# [Latent factor model](https://www.youtube.com/watch?v=E8aMcwmqsTg&list=PLLssT5z_DsK9JDLcT8T62VtzwyW9LNepV&index=55)
# 
# Now enough with this discussion .... let's dive into learning.....
# 
#  Since the dataset we used before did not have userId(which is necessary for collaborative filtering) let's load another dataset. We'll be using the Surprise library to implement SVD.

# In[ ]:


from surprise import Reader, Dataset, SVD
reader = Reader()
ratings = pd.read_csv('../input/for-collaborative/ratings_small.csv')
ratings.head()


# Here rating is given in the scale of 5
# 
# Now we will make sure reading the dataset with reader to work in surprise library .... so first do that

# In[ ]:


data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)


# create an svd object to call the fit method which makes rmse less and then transform to two matrices 

# In[ ]:


svd = SVD()


# Let us now train on our dataset and arrive at predictions.

# In[ ]:


trainset = data.build_full_trainset()
svd.fit(trainset)


# Now let check how it works:-

# In[ ]:


ratings[ratings['userId'] == 1]#actual ratings


# In[ ]:


svd.predict(1, 31)


# so predicted is 2.4 and the actual is 2.5 so we can say that it works significantly well...

# **Conclusion**
# 
# We create recommenders using demographic , content- based and collaborative filtering. While demographic filtering is very elemantary and cannot be used practically, Content based recommender are used in the filtered searching based on some filter we asked for like action genre movies like that and the collaborative recommender systems are more personalized and those are used for improving the user experience. This model was very baseline and only provides a fundamental framework to start with.And you have all the knowledge required to build to recommender system...

# **References**
# 
# [https://hackernoon.com/introduction-to-recommender-system-part-1-collaborative-filtering-singular-value-decomposition-44c9659c5e75](https://hackernoon.com/introduction-to-recommender-system-part-1-collaborative-filtering-singular-value-decomposition-44c9659c5e75)
# 
# [tps://www.kaggle.com/rounakbanik/movie-recommender-systems](https://www.kaggle.com/rounakbanik/movie-recommender-systems)
# 
# please checkout the MIT courseware videos on SVD,Latent Factor [MIT Latent Factor](https://www.youtube.com/watch?v=E8aMcwmqsTg&list=PLLssT5z_DsK9JDLcT8T62VtzwyW9LNepV&index=55)
# 

# If you find it helpful click an upvote and leave your suggestions below...Thank you
