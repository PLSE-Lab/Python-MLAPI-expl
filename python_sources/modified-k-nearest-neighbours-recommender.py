#!/usr/bin/env python
# coding: utf-8

# Our task is to build a recommender system for Netflix's huge library of movies and TV shows! How hard could it be...

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import math


# In[ ]:


df = pd.read_csv("../input/netflix-shows/netflix_titles.csv")
df.head()


# In[ ]:


print(df.columns)


# We examine the number of NA values. From the results show below they are mostly in the director, cast and country column. We drop these, which unfortunately cuts our dataset size by a hefty margin.

# In[ ]:


original_len = len(df)
dropped_df = df.dropna()
dropped_len = len(dropped_df)
print("% of rows with missing values: " + str((original_len - dropped_len) / dropped_len * 100) + '%')
print()
print("Number of null values in each column: ")
print(df.isnull().sum())
df = dropped_df


# Our strategy is to build a recommender system using a modification of the k-Nearest-Neighbours algorithm on columns that we think would be the most useful. First off, recommendations probably won't be based off of features like runtime, show_id, the date added to Netflix, so we drop irrelevant columns. We also drop the description, but ideally we would pick up keywords from the description using natural language processing to use later.

# In[ ]:


df.drop(columns=['show_id', 'date_added', 'duration', 'description'], inplace=True)


# We'll encounter some trouble with the director, cast and listed_in features, since the datapoints are comma-separated values of all the directors, cast members and genres the movie or show is listed in.
# 
# We'll use a **bag of words** technique on these columns. First off only keep the first three values in each of these columns, and discard the rest. We will then assemble a bag of words for each record made up from words in these features. We can calculate the 'distance' between these bags of words later, using cosine similarities.

# In[ ]:


df['country'] = df['country'].map(lambda x: x.split(',')[0])

bag_of_words_data = ['director', 'cast', 'listed_in']

for col in bag_of_words_data:
    df[col] = df[col].map(lambda x : x.lower().replace(' ', '').split(',')[:3])

df['bag_of_words'] = ''
for i, row in df.iterrows():
    words = [' '.join(row[col]) for col in bag_of_words_data]
    df.loc[i, 'bag_of_words'] = ' '.join(words)
    
df.drop(columns=bag_of_words_data, inplace=True)
df.head()


# We one-hot encode the remaining categorical columns (type, country and rating).

# In[ ]:


df = pd.get_dummies(df, columns=['type', 'rating', 'country'])
df.reset_index(drop=True, inplace=True)


# We use SKLearn's CountVectoriser to create a matrix of word counts out of all the bags of words in the dataset. From this we can generate a cosine similarity matrix, effectively generating the similarity between each record and every other record in terms of their bags of words (will be between 0 and 1). We convert similarities into dissimilarities by subtracting each similarity from 1.
# 
# We could have used a TF-IDF Vectoriser (term-frequency to inverse-document-frequency), however this would give less weight to words which occur more often in the bags of words - since the bags contain genre categories which are repeated a lot, this is not what we want!

# In[ ]:


vectorizer = CountVectorizer()
count_matrix = vectorizer.fit_transform(df['bag_of_words'])

similarities = cosine_similarity(count_matrix, count_matrix)
dissimilarities = 1 - similarities


# We use a combination of the Euclidean distance between records in terms of their features in the dataset, plus their dissimilarities in terms of bags of words, to formulate an overall distance metric for our nearest neighbours algorithm. Our custom distance metric can be expressed in the following equation:
# 
# $$ \begin{align} \textrm{distance} & = \textrm{euclidean_distance}_{type, country, rating} \\ & + \alpha \textrm{ euclidean_distance}_{release\_year} \\ & + \beta \textrm{ cosine_dissimilarity}_{bag\_of\_words} \end{align} $$
# 
# The component of the Euclidean distance involving the release year column will be much larger than for the type, country and rating columns, since the latter are one-hot encoded, so we will scale this down by some factor $ \alpha $ (0.1 by default). We can also control the influence that the bag of words containing genre, cast and director will have with the $ \beta $ parameter (2 by default).

# In[ ]:


def euclidean_distance(row1, row2, release_year_weighting=0.1):
    row1_features = np.array([row1[col] for col in df.columns if col != 'title' and col != 'bag_of_words' and col != 'release_year'])
    row1_features = row1_features.astype(np.int16)
    row2_features = np.array([row2[col] for col in df.columns if col != 'title' and col != 'bag_of_words' and col != 'release_year'])
    row2_features = row2_features.astype(np.int16)
    diffs = np.subtract(row1_features, row2_features)

    release_year_diff = release_year_weighting * (row1['release_year'] - row2['release_year'])
    diffs = np.append(diffs, [release_year_diff])

    return math.sqrt(np.sum([diff ** 2 for diff in diffs]))

def total_distance(row1_index, row2_index, bag_of_words_weighting=2):
    row1 = df.iloc[row1_index]
    row2 = df.iloc[row2_index]
    distance = euclidean_distance(row1, row2)
    distance += bag_of_words_weighting * dissimilarities[row1_index][row2_index]
    return distance


# We create a recommend function which will run the k nearest neighbours algorithm on a particular Netflix title and return the results (default k = 10).

# In[ ]:


def recommend_for(title, num_recommendations=10):
    all_titles = df['title']
    title_instances = all_titles[all_titles == title]
    if title_instances.empty:
        print("Sorry! We can't seem to find that movie in our collection")
        return
    curr_index = all_titles[all_titles == title].index[0]

    distances = list()
    for i, row in df.iterrows():
        distances.append((row, total_distance(curr_index, i)))
    distances.sort(key=lambda tup: tup[1])
    results = list(map(lambda tup: tup[0]['title'], distances[1:num_recommendations+1]))
    print("After watching " + title + ", we recommend: ")
    for res in results:
        print(res)
    print()


# In[ ]:


recommend_for("The Battle of Midway")
recommend_for("You")
recommend_for("The Perfect Date")


# The recommender generates some cool stuff! If the results look off we can always tune the parameters within our distance-based model. One unfortunate downside is that due to a lazy k-nearest-neighbours approach the recommender is very slow - using an eager learning algorithm would definitely fix this.

# In[ ]:




