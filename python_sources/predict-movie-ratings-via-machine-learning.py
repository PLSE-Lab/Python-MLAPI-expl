#!/usr/bin/env python
# coding: utf-8

# # Predict Movie Ratings using Machine Learning

# ## Project overview
# To see a more robust version of this project, please check out [my github repository][github]. I use Amazon Sagemaker to train a model using XGBoost, which gave me more accurate predictions, and an R^2 score of 0.829.
# 
# For this project, I will be using the [5000 movies dataset][db], and supervised learning, to predict the user score for a film. The motivation behind this project comes from my [benchmark model][kernel], by Ashwini Swain; here, Swain makes accurate movie-score predictions (average accuracy of about 98.5% among three movies (Godfather Part 3, Minions, Rocky Balboa)) by comparing films' genres, actors, directors, and keywords. Then, to predict the average, Swain does the following:
# - Calculate the cosine similarity among other movies
#   - First by creating binary-arrays for the lists of genres, actors, directors, and keywords
#   - Then, by using scipy's cosine similarity method to compare the arrays with those of other movies
# - Takes the top 10 similar movies
# - Calculates the average of these 10 movies' user scores
# - Uses that average as the prediction for the subject movie's score
# I used his algortihm with 10 random movies to get an average accuracy of about 87%. Because it takes too long to predict the score of all movies in the dataset, it must be noted that this score may or may not represent the overall accuracy of all movie predictions
# 
# For my project, instead of using a cosine-similarity algorithm, I will use machine learning to predict a movie's score. To do this, I will: 
# - create a decimal representation of the features: genres, actors, directors, keywords, and production companies
#   - Like Swain has done, I will create a binary array for each feature. Then, I will analyze the array like points in a number line
#     - Looking at the points, I can identify a point that best describes the distribution of 1's, and use that as a decimal representation of the feature
# - normalize the decimal values to be within [0,1]
# - Split data into training and testing sets
#   - 70% for training, 30% for testing
# - Train a regression-based model using Scikit learn's Bayesian Ridge
# - Make predictions and evaluate using an R^2 score
# 
# I will only use movies that fall under this criteria
#  - Rating of movie is greater than 0.0 (0,0 rating corresponds to movies with 0 or 1 votes, so there's no telling if the rating is an accurate representation of the movie)
#  - Movie has at least one of the following features: cast, crew, production companies
# [//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
# 
# 
#    [kernel]: <https://www.kaggle.com/ash316/what-s-my-score>
#    [db]: <https://www.kaggle.com/tmdb/tmdb-movie-metadata#tmdb_5000_movies.csv>
#    [github]: <https://github.com/b-raman/movie-rating-predictor>

# ## Setup
# ### Import Modules

# In[ ]:


import io
import os
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 

import json #converting JSON to lists for dataframe
import warnings
warnings.filterwarnings('ignore')
import base64
import codecs
from IPython.display import HTML

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


movie1 = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_movies.csv")
movie2 = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_credits.csv")

movies = movie1.merge(movie2,left_on='id',right_on='movie_id',how='left')# merging the two csv files


# ### Proof that movies with 0.0 rating only have vote count of 0 or 1

# In[ ]:


counts = movies[(movies['vote_average']==0)]['vote_count'] # get vote counts for all movies that have a rating of 0.0

print("Unique vote counts for movies with 0.0 rating")
for u in set(counts):
    print(u)


# ### Remove 0.0 rated movies from dataframe

# In[ ]:


movies = movies[(movies['vote_average']!=0)]


# Quick look at movies

# In[ ]:


movies.sample(5)


# ### Step 1: Format dataframe
# Features: **genres, keywords, cast, crew, and production companies** are all in JSON format. Convert to simple list via following function

# In[ ]:


def to_list(df, feature_names_list): #df: dataframe, feature_names: list of all features to convert from JSON to list
    for feature_name in feature_names_list:
        print("Current:", feature_name)
        #STEP 1: convert JSON format to a list
        df[feature_name]=df[feature_name].apply(json.loads)
        #Two cases here: Feature is crew, or feature is not crew
        if feature_name == 'crew': #if crew, due to large size, want to limit to most influential members: director, editor, cinematographer, screenplay, and composer
            for index,i in zip(df.index,df[feature_name]):
                feature_list_1=[]
                limit = 10
                if len(i) < 10:
                    limit = len(i)
                for j in range(limit): #top 10 crew members
                    feature_list_1.append((i[j]['name'])) # the key 'name' contains the name of the a sub-feature (ex: sci-fi in genres)
                df.loc[index,feature_name]= str(feature_list_1)
        
        elif feature_name == 'cast': #Another special case. Only want top 5 cast members (most infulential)
            for index,i in zip(df.index,df[feature_name]):
                feature_list_1=[]
                if len(i) > 5:
                    limit = 5
                else:
                    limit = len(i)
                for j in range(limit): #top 5 (JSON format already has this sorted)
                    feature_list_1.append((i[j]['name']))
                df.loc[index,feature_name]= str(feature_list_1)
        else:    
            for index,i in zip(df.index,df[feature_name]):
                feature_list_1=[]
                for j in range(len(i)):
                    feature_list_1.append((i[j]['name']))
                df.loc[index,feature_name]= str(feature_list_1)
    
        #STEP 2: clean up and transform into unsorted list
        df[feature_name] = df[feature_name].str.strip('[]').str.replace(' ','').str.replace("'",'')
        df[feature_name] = df[feature_name].str.split(',')
        
        #STEP 3: Sort list elements
        for i,j in zip(df[feature_name],df.index):
            features_list_2=i
            features_list_2.sort()
            df.loc[j,feature_name]=str(features_list_2)
        df[feature_name]=df[feature_name].str.strip('[]').str.replace(' ','').str.replace("'",'')
        lst = df[feature_name].str.split(',')
        if len(lst) == 0:
            df[feature_name] = None
        else:
            df[feature_name]= df[feature_name].str.split(',')
    return df


# In[ ]:


movies = to_list(movies, ['genres', 'keywords', 'production_companies', 'cast', 'crew']) #function call


# ### Step 2: Remove movies with empty features
# 
# If a movie has no information on cast, crew, and production companies, the movie does not match the criteria for a valid datapoint (There's nothing to take away from a movie with no cast, crew, or production company). So, remove movie(s) from dataset

# In[ ]:


to_drop = []
for i in movies.index:
    if (movies['production_companies'][i] == [''] and movies['cast'][i] == [''] and 
        movies['crew'][i] == ['']):
        to_drop.append(i)
print('Dropping', str(len(to_drop)), 'movies.')
movies = movies.drop(to_drop, axis = 0)


# In[ ]:


movies.shape[0]


# #### Remove un-needed feature-types in dataframe

# In[ ]:


movies_shortened = movies[['id','original_title','genres','cast', 'crew', 'production_companies', 'keywords', 'vote_average']]


# Quick look at shortened and formatted movies

# In[ ]:


movies_shortened.sample(10)


# ### Looking at distribution of Movie ratings:
# - Histogram of distribution
# - minimum of ratings
# - maximum of ratings
# - mean and variance

# In[ ]:


plt.subplots(figsize=(12,10))
n, bins, patches = plt.hist(movies_shortened['vote_average'], 30, density=1, facecolor='g', alpha=0.75)

plt.xlabel('Vote_average')
plt.ylabel('Occurence')
plt.title('Distribution of voter average')
plt.grid(True)
plt.show()
print("Minimum of Ratings:", round(min(movies_shortened['vote_average']),2))
print("Maximum of Ratings:", round(max(movies_shortened['vote_average']),2))
print("Average of Ratings:", round(np.mean(movies_shortened['vote_average']),2))
print("Variance of Ratings:",round(np.var(movies_shortened['vote_average']),2))


# ## Feature Engineering: Turn lists of features into numerical representations

# ### Step 1: Identify all unique sub-feature for each feature (ex: all unique cast members in the cast category)
# #### Organize sub-features to lowest-rating association to highest-rating association
# By organizing features in this order, the numerical representation will then also describe the quality of features associated with the movie
# - higher quality feature values should correlate with higher-rated movies, and vice versa

# In[ ]:


def generate_list(df, feature_name): #create a list of all unique feature values
    #Step 1: track all ratings associated with each feature in a dictionary
    feature_dict = {}
    for index, row in df.iterrows():
        feat = row[feature_name]
        for sub_feat in feat:
            if sub_feat not in feature_dict:
                feature_dict[sub_feat] = (df['vote_average'][index], 1) #
            else:
                feature_dict[sub_feat] = (feature_dict[sub_feat][0] + (df['vote_average'][index]), feature_dict[sub_feat][1] + 1)
    #Step 2: calculate average ratings for each feature
    for key in feature_dict:
        feature_dict[key] = feature_dict[key][0]/feature_dict[key][1] #average of all vote_averages
       
    #Step 3: create and sort a list of tuples (dictionary value, key)
    lst = list()
    for name in feature_dict:
        lst.append((feature_dict[name],name))
    lst = sorted(lst)
    #step 4: create a list of only the feature names, from lowest rating to highest rating
    feature_list = list()
    ratings_list = list()
    for element in lst:
        feature_list.append(element[1])
        ratings_list.append(element[0])
    
    #get the variance of the ratings. This is helpful for determining the usefulness of the information (to be displayed in below plot)
    var = round(np.var(ratings_list),3)
    
    #before returning the list, do a quick visualization to show that generate_list works
    fig, ax = plt.subplots(figsize=(6,5))
    if feature_name != 'genres':
        n = 50 # sample at intervals of n
    else:
        n = 1
    X = [] #sample for associated movie(s) rating average
    Y = [] #sample for feature names
    for i in range(0, len(feature_list) - 1, n):
        X.append(ratings_list[i])
        Y.append(feature_list[i])
    
    y_pos = np.arange(len(Y))
    ax.barh(y_pos, X, align='center')
    #ax.set_yticklabels(Y)
    ax.invert_yaxis()  # labels read top-to-bottom
    
    ax.set_xlabel('Overall average movie ratings')
    ax.set_ylabel(feature_name + ' sample list index')
    ax.set_title(feature_name + ' to associated movie(s) performance (' + str(int(len(feature_list)/n)) + ' samples), variance: ' + str(var))
    
    plt.show()
    
    return feature_list


# ### Create lists for each feature

# In[ ]:


genres_list = generate_list(movies_shortened, 'genres')


# In[ ]:


cast_list = generate_list(movies_shortened, 'cast')


# In[ ]:


crew_list = generate_list(movies_shortened, 'crew')


# In[ ]:


prod_companies_list = generate_list(movies_shortened, 'production_companies')


# In[ ]:


keywords_list = generate_list(movies_shortened, 'keywords')


# ### Analysis
# 
# Judging by the above variances, it's safe to say that **genres** will not be useful for predicting a movie's vote-average
# The features **cast, crew, production companies, and keywords** have high variances, which will be useful

# In[ ]:


movies_shortened = movies_shortened[['id', 'original_title', 'cast', 'crew', 'production_companies', 'keywords','vote_average']]


# ### Step 2: Create a binary representation for each feature
# 
# ##### Using the lists created, create binary arrays that indicated whether or not feature_name can be found in this movie
# 
# note: each array represents a feature associated with movies with lowest average ratings to highest average ratings
# - this is useful because we can use the array as a gauge for how well the features track record in movies are

# In[ ]:


def calculate_bin_array(this_list, all_features):
    bin_list = []
    for element in all_features:
        if element in this_list:
            bin_list.append(1)
        else:
            bin_list.append(0)
    return bin_list


# In[ ]:


movies_shortened['cast'] = movies_shortened['cast'].apply(lambda x: calculate_bin_array(x, cast_list))


# In[ ]:


movies_shortened['crew'] = movies_shortened['crew'].apply(lambda x: calculate_bin_array(x, crew_list))


# In[ ]:


movies_shortened['production_companies'] = movies_shortened['production_companies'].apply(lambda x: calculate_bin_array(x, prod_companies_list))


# In[ ]:


movies_shortened['keywords'] = movies_shortened['keywords'].apply(lambda x: calculate_bin_array(x, keywords_list))


# In[ ]:


movies_shortened.sample(10)


# ### Look at distribution of 1's in a number-line format

# In[ ]:


def plot_bin(mov):
    cast_bin = mov[2]
    cast_index = []
    # create arrays of indeces where bin number is one
    for i in range(len(cast_bin)):
        if cast_bin[i] == 1:
            cast_index.append(i)
    
    crew_bin = mov[3]
    crew_index = []
    for i in range(len(crew_bin)):
        if crew_bin[i] == 1:
            crew_index.append(i)
    
    prod_bin = mov[4]
    prod_index = []
    for i in range(len(prod_bin)):
        if prod_bin[i] == 1:
            prod_index.append(i)
    
    keywords_bin = mov[5]
    keywords_index = []
    for i in range(len(keywords_bin)):
        if keywords_bin[i] == 1:
            keywords_index.append(i)
    
    font = {'family': 'serif',
        'color':  'red',
        'weight': 'normal',
        'size': 10,
        }
    
    fig, ax = plt.subplots(4,1,figsize=(5,1))
    plt.subplots_adjust(hspace = 5)
    ax[0].scatter(cast_index, np.zeros_like(cast_index), vmin=-2)
    ax[0].set_title('Cast', loc = 'left', fontdict=font)
    ax[0].set_xlim(0,len(cast_bin))
    ax[0].set_yticks([])
    ax[0].set_xticks([])
    
    ax[1].scatter(crew_index, np.zeros_like(crew_index), vmin=-2)
    ax[1].set_title('Crew', loc = 'left', fontdict=font)
    ax[1].set_xlim(0,len(crew_bin))
    ax[1].set_yticks([])
    ax[1].set_xticks([])
    
    ax[2].scatter(prod_index, np.zeros_like(prod_index), vmin=-2)
    ax[2].set_title('Production companies', loc = 'left', fontdict=font)
    ax[2].set_xlim(0,len(prod_bin))
    ax[2].set_yticks([])
    ax[2].set_xticks([])
    
    ax[3].scatter(keywords_index, np.zeros_like(keywords_index), vmin=-2)
    ax[3].set_title('Keywords', loc = 'left', fontdict=font)
    ax[3].set_xlim(0,len(keywords_bin))
    ax[3].set_yticks([])
    ax[3].set_xticks([])


# In[ ]:


movies_sample = movies_shortened.sample(5)


# In[ ]:


print('Movie: ' + movies_sample.iloc[0][1] + '\nRating: ' + str(movies_sample.iloc[0][-1]) + '\n')
plot_bin(movies_sample.iloc[0])


# In[ ]:


print('Movie:' + movies_sample.iloc[1][1] + '\nRating: ' + str(movies_sample.iloc[1][-1]) + '\n')
plot_bin(movies_sample.iloc[1])


# In[ ]:


print('Movie:' + movies_sample.iloc[2][1] + '\nRating: ' + str(movies_sample.iloc[2][-1]) + '\n')
plot_bin(movies_sample.iloc[2])


# In[ ]:


print('Movie:' + movies_sample.iloc[3][1] + '\nRating: ' + str(movies_sample.iloc[3][-1]) + '\n')
plot_bin(movies_sample.iloc[3])


# In[ ]:


print('Movie:' + movies_sample.iloc[4][1] + '\nRating: ' + str(movies_sample.iloc[4][-1]) + '\n')
plot_bin(movies_sample.iloc[4])


# #### Analysis
#  - Movies with lower ratings have features leaning towards the left, while movies with higher ratings have features leaning to the right. 
#  - This shows that there's a relationship between features and ratings if feature names is organized from lowest movie rating associations to highest movie rating associations

# ### Step 3: Find concentration points in each array
# Find areas where numbers are grouped, and identify points that resemble the centers of binary distribution
# 
#     Generic example: [1110001111100101] -> [0300000050000020] -> [(1,3), (8,5), (14,2)] tuple[0] is the index of concentration, tuple[1] is the number of 1's about index

# In[ ]:


def split_arr(arr, n_splits): 
      
    # looping till length l 
    for i in range(0, len(arr), n_splits):  
        yield arr[i:i + n_splits] 

def find_concentration(arr, n = 100): # n is the number of concentration points to find
    #seperate array into batches
    batches = list(split_arr(arr,int(len(arr)/n)))
    concentrations = []
    for i in range(len(batches)):
        point = 0
        num_ones = 0
        for j in range(len(batches[i])):
            if batches[i][j] == 1:
                point += j + (i * int(len(arr)/n)) # adding correction for batches
                num_ones += 1
        if num_ones > 0:
            point = point/num_ones
            concentrations.append((point,num_ones))
    return concentrations


# In[ ]:


def to_concentrations(df, feature_names):
    for feature_name in feature_names:
        print('feature: ', feature_name)
        df[feature_name] = df[feature_name].apply(lambda x: find_concentration(x))
    return df


# In[ ]:


movies_shortened = to_concentrations(movies_shortened, ['cast', 'crew', 'production_companies', 'keywords'])


# In[ ]:


movies_shortened.sample(10)


# ### Step 3.1: Find a decimal value that represents the concentration points
# The point will represent the weighted average of all points of concentration
# The weight is the number of ones for each concentration point

# In[ ]:


def w_avg(arr):
    weight = 0 #weight
    s = 0 # position*weight
    for element in arr:
        s += (element[0] * element[1])
        weight += element[1]
    return s/weight #weighted average


# In[ ]:


def to_weighted_avg(df, feature_names):
    for feature_name in feature_names:
        print('Current: ', feature_name)
        df[feature_name] = df[feature_name].apply(lambda x: w_avg(x))
    return df


# In[ ]:


movies_shortened = to_weighted_avg(movies_shortened, ['cast', 'crew', 'production_companies', 'keywords'])


# In[ ]:


movies_shortened['vote_average'] = movies['vote_average']


# In[ ]:


movies_shortened.sample(10)


# ### Step 4: Normalize the features

# First, make a dataframe to isolate the features

# In[ ]:


feat_df = movies_shortened[['cast', 'crew', 'production_companies', 'keywords']] #extract only features from df, and scale


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
feat_scaled = pd.DataFrame(scaler.fit_transform(feat_df.astype(float)))
feat_scaled.index = feat_df.index
feat_scaled.columns = feat_df.columns

#Seperate dataframe for target
target_df = pd.DataFrame()
target_df['ratings'] =  movies_shortened['vote_average']


# In[ ]:


feat_scaled.sample(10)


# ### Visualising Features in a Scatterplot

# In[ ]:


fig, ax = plt.subplots(2,2, figsize=(24,20))

ax[0,0].scatter(target_df['ratings'], feat_scaled['cast'], facecolor='blue')
ax[0,0].set_xlabel('rating')
ax[0,0].set_ylabel('cast normalized')
ax[0,0].set_title('cast')

ax[1,0].scatter(target_df['ratings'], feat_scaled['crew'], facecolor='green')
ax[1,0].set_xlabel('rating')
ax[1,0].set_ylabel('crew normalized')
ax[1,0].set_title('crew')

ax[0,1].scatter(target_df['ratings'], feat_scaled['production_companies'], facecolor='red')
ax[0,1].set_xlabel('rating')
ax[0,1].set_ylabel('production companies normalized')
ax[0,1].set_title('Production Companies')

ax[1,1].scatter(target_df['ratings'], feat_scaled['keywords'], facecolor='orange')
ax[1,1].set_xlabel('rating')
ax[1,1].set_ylabel('keywords normalized')
ax[1,1].set_title('keywords')

fig.suptitle("Corrlation between a movie's features and its rating")


# As you can see, there's a clear correlation between the features and the ratings
# The straight lines in figures for keywords and production companies represents the absence of keywords and production companies for certain movies

# ### Split Data into testing and training
# Will be splitting trainting : testing : validation -> (0.7) : (0.15) : (0.15)

# In[ ]:


from sklearn.model_selection import train_test_split
def train_test_val_split(df_feat, df_target, train_frac):
    train_features, test_features, train_target, test_target = train_test_split(df_feat, df_target, test_size = train_frac) #splitting training from rest of the dataset
    return (train_features, train_target), (test_features, test_target)


# In[ ]:


(features_train, target_train), (features_test, target_test) = train_test_val_split(feat_scaled, target_df,0.7)


# In[ ]:


target_train.head()


# ## Creating the model

# #### Import Scikit Learn's [Bayesian Ridge][bayes] Regressor
# 
# Bayesian ridge is one of many regression models offered by Scikit Learn. I'm choosing this model because it's ideal for dealing with data containing multiple outliers (movies with ratings inconsistent with its features)
# 
# 
# [//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
# 
# 
#    [bayes]: <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge>

# In[ ]:


from sklearn.linear_model import BayesianRidge


# #### Create and train model

# In[ ]:


reg = BayesianRidge()
reg.fit(features_train.values, target_train)


# #### Make predictions with features_test

# In[ ]:


target_pred = reg.predict(features_test.values)


# ## Evaluation
# #### Plot predictions vs test ratings

# In[ ]:


plt.axis([0,10,0,10])
plt.scatter(target_test, target_pred)

index_arr = [n for n in range(11)]
plt.plot(index_arr,'r--')             
plt.xlabel("Movie Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Movie ratings vs Predicted ratings")


# #### Get R^2 score
# 
# An [r^2 score][r2] is ideal for regression models because it gauges how well the variance of movie ratings can be explained by the features used (cast, crew, production companies, and keywords)
# 
# [//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
# 
# 
#    [r2]: <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html>
# 

# In[ ]:


from sklearn.metrics import r2_score

score = r2_score(target_test, target_pred)

print("R^2 Score for predictions:", score)


# ## Conclusion
# 
# Though 0.7997 shows that the variance of ratings in can be explained by movie features, it's not as high as I'd hoped it would be.
# Using Amazon Sagemaker, I used a more powerful algorithm, XGBoost, to create a better predictor. The resultant R^2 score I got from that was 0.829!
# 
# The project can be found [here][xgb] if you are interested.
# 
# Please let me know what you thought of my project!
# 
# Thanks to Ashwini Swain for his kernel on predicting movie ratings.
# 
# [//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
# 
# 
#    [xgb]: <https://github.com/b-raman/movie-rating-predictor>
#     
