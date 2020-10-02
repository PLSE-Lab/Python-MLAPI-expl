# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("/kaggle/input/movie_dataset.csv")

print(df.columns)

#select features
features = ["keywords","cast","genres","director"]

#create a column in df which combines all selected features
for feature in features:
    df[feature] = df[feature].fillna('')



def combined_features(row):
    try:
        return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
    except:
        print("Error:",row)

df["combined_features"] = df.apply(combined_features, axis=1)


print(df["combined_features"].head())


#create count matrix from this new combined column
cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])


#compute the cosine similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)

print(cosine_sim.shape)


###### Helper Function
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]


def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

#Get index of the movie from its title
movie_user_likes = "Avatar" 
movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(cosine_sim[movie_index]))

#geta list of similar movies in descending oerder of similarity score

sorted_similar_movies = sorted(similar_movies, key= lambda x:x[1], reverse=True)
print(sorted_similar_movies)

#get titles of some movies
i=0
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0]))
    i+=1
    if i>20:
        break
