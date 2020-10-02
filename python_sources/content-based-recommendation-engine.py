# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Importing Dataset, download in kaggle site : https://www.kaggle.com/tmdb/tmdb-movie-metadata
credit_data = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')
movie_data = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')

credit_data.head()

movie_data.head()

credit_data.describe()

credit_data.info()

movie_data.describe()

movie_data.info()

print("Shape of Credit :  {}".format(credit_data.shape))
print("Shape of Movie :  {}".format(movie_data.shape))

credit_col_rename = credit_data.rename(index=str,columns={'movie_id':'id'})

credit_col_rename

#Merge those two dataset using ID as foreign key
movie = movie_data.merge(credit_col_rename,on='id')

movie

movie.columns

un_col=['homepage','title_x','title_y','status','production_companies']
movie[un_col]

#Lets drop the unnecessary columns
movie = movie.drop(columns=['homepage','title_x','title_y','status','production_companies'])

movie.head()

movie.info()

## Content Based Recommendation System

movie.head(1)['overview']

### Using Tf-IDF Vectorizer to formulate vectorization matrix 

from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(
    min_df=3,
    max_features=None,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1,3),
    stop_words='english'
)

movie['overview']  = movie['overview'].fillna('')

#Fitting the TF-IDF on 'overview' text
tf_matrix = tf.fit_transform(movie['overview'])

tf_matrix

tf_matrix.shape

from sklearn.metrics.pairwise import sigmoid_kernel

#Compute sigmoid kernel
sig = sigmoid_kernel(tf_matrix,tf_matrix)

sig[0]

#Reverse mapping of indices and movie titles
indices = pd.Series(movie.index,index=movie['original_title']).drop_duplicates()

indices

indices['Avatar']

sig[0]

sorted(list(enumerate(sig[indices['Avatar']])),key=lambda x:x[1],reverse=True)

#Finally lets make a function to get top 10 recommendation of movies
def give_rec(title,sig=sig):
    #Get the corresponding to original_title
    index = indices[title]
    
    #Get the pairwise similiarity scores
    sig_scores = list(enumerate(sig[index]))
    
    #Sort the movies
    sig_scores = sorted(sig_scores,key=lambda x :x[1],reverse=True)
    
    #Score of 10 most similar movies
    sig_scores = sig_scores[1:11]
    
    #Movie indices
    movie_indices = [i[0] for i in sig_scores]
    
    #Top 10 most similar movies
    return movie[['original_title','vote_average']].iloc[movie_indices]

#Lets test it out
give_rec('Avatar')

