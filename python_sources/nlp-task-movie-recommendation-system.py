#!/usr/bin/env python
# coding: utf-8

# I have tried my best to explained all the steps that I have performed in the notebook and my approach for the task as well. 
# 
# I always aim to create an **accurate solution ** and not just a solution. Here too I have tried to achieve the same!!

# Importing necessary libraries

# In[ ]:


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


# Here we have several dataset csv files. Out of all the files, I have taken into consideration two important files:
# 1. The main dataset i.e movies_metadata.csv
# 2. The credits.csv file which conatins details about the cast and the crew
# 
# Finally, we read the mentioned dataset i.e csv files

# In[ ]:


df1=pd.read_csv('../input/the-movies-dataset/credits.csv')
df2=pd.read_csv('../input/the-movies-dataset/movies_metadata.csv')


# 
# 
# Refer the following link for information about the dataset:
# 
# https://www.kaggle.com/rounakbanik/the-movies-dataset#movies_metadata.csv[](http://)
# 
# Printing first 5 entries of movies_metadata.csv

# In[ ]:



df2.head()


# In[ ]:


df2['id']


# In[ ]:


df1['id']


# Removing some redundant entries from the movies_metadata.csv

# In[ ]:


df2 = df2[df2.id!='1997-08-20']
df2 = df2[df2.id!='2012-09-29']
df2 = df2[df2.id!='2014-01-01']


# In[ ]:


df1.id


#  Converting "id" column into integer type so we can finally combine both the dataset files[](http://) as a single dataset

# In[ ]:


df2['id'] = df2['id'].astype(int)


# In[ ]:


df2.id


# Now we merge both the dataset

# In[ ]:


df2=df2.merge(df1, on='id')


# These are the columns we have finally in our dataset

# In[ ]:


df2.columns


# Now we can start with building our recommender system. There are three main approaches in recommender system.
# 1. Demographic Filtering
# 2. Content based Filtering
# 3. Collaborative Filtering
# 
# Here I use content based filtering technique. In content based filtering, we recommend user, similar movies based on what users have seen in the past or what users like. To achieve this, we can use various parameters like movie title, cast, genre, movie overview, votes etc

# ### **Approach:**
# 
# **So here, I aim to create a generalized solution by taking into account all the *important parameters* for a particular movie.
# This will give the user a much more personalized experience and user can get a recommnendation according to his requirements. **

# ### Main Task:
# 
# #### I  first tried to build a system which can recommend the movies according to the movie description/overview
# 
# #### **Here I use the *overview data column* for building this system**
# 

# In[ ]:


df2['overview'].head()


# I calculate the **similarity scores**  for the overview of each movie entry in the dataset 
# <br>
# 
# for that I  use a text encoding technique known as **word_to_vec** in order to convert each overview into numeric embeddeings. This technique is used to convert textual data into numeric vectors

# Then we compute **Term Frequency-Inverse Document Frequency (TF-IDF) **vectors for each overview.

# This will give us a relative frequency of word in the document.
# To achieve this, I use scikit-learn which gives  a built-in TfIdfVectorizer class that produces the TF-IDF matrix

# In[ ]:


#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df2['overview'] = df2['overview'].fillna('')


# The shape of the tfid_matrix gives us the number of distinct words used to define our entries of 45538 movies. We have 75827 distinct english words used to describe the overview of the movies 

# Now to calculate similarity score, I did a bit of research on which are the best methods for the same. I found several techniques like euclidean, the Pearson and the cosine similarity scores. I will be using cosine similarity scores to find the similarity between two movies
# <br>
# One advantage of cosine similarity score is that it is independent of magnitude and is relatively easy and fast to calculate. 

# In[ ]:


from sklearn.metrics.pairwise import linear_kernel




#  As we can see that we have a huge dataset and due to lack of computational power of this kernel, the kernel keeps on dying (we just have 16GB RAM) so we cannot compute the dot product of all the 45000 movies. 
#  <br>
#  
# So instead I have taken a random part of data in order to generate the cosin values

# After experimenting with many values, I found that kaggle kernel can bear upto 35000 entries of movies dataset. Anything greater and the kernel ram exceds 16GB!! 
# <br>
# So now taking a random part of dataset to generate cosine values.

# In[ ]:


import random
ran = random.randint(25000, 30000)
df3 = df2.head(ran)
tfidf_matrix = tfidf.fit_transform(df3['overview'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape


# In[ ]:


cosine_sim =  linear_kernel(tfidf_matrix, tfidf_matrix, True)


# In[ ]:


indices = pd.Series(df3.index, index=df3['title']).drop_duplicates()


# Now I create a *get_recommendation* function which accpets a movie title from the user and recommends similar movies to the user based on the title by taking the cosine simalarity scores of the most similar movies.
# <br>
#  This function will recommend the 10 most similar movies

# In[ ]:


def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df3['title'].iloc[movie_indices]


#  Time to Test my code XD

# In[ ]:


get_recommendations('Assassins')


# **WOAHH !!!!**  I gave input as a crime movie and it was able to succesfully recommend me set of similar crime movies

# Another try for any romantic movie!!

# In[ ]:


get_recommendations('Carrington')


# Yeah!! We get recommendation of similar romantic movies

# **Glad, that my code is able to succesfully recommend movies based on movie title and overview**

# Although the system is able to recommend similar movies, but still the quality of recommendation is not that great as we have only taken movie overview int account. 

# ### Further enhancing the model and approach:

# As I mentioned before that I always try my best to build an accurate solution. I therby referred a few tutorials on recommendation systems. 

# So, I am further trying to use other properties like movie genre, ratings, keywords, crew etc 
# 
# **Taking important features like movie genres, caste of the movie, keywords etc will help to build a much more accurate recommender system**

# In[ ]:


df4=pd.read_csv('../input/the-movies-dataset/keywords.csv')


# First we read the keywords.csv dataset

# In[ ]:


df4.head()


# In[ ]:


df2 = df2.merge(df4, on='id')


# We finally merge keywords dataset with our original dataset

# In[ ]:


df2.head()


# Now that we have our merged dataset, so I can extract important features like genres, keywords, cast etc

# In[ ]:


from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)


# In[ ]:


df2.head()


# Now I wrote some functions to clean and extract  important features like directors, genres, and keywords from the data

# In[ ]:


#function to return director name from crew column
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# In[ ]:


#function to get names of genres, cast and keywords from the data
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


# Print the new features of the first 3 films
df2[['title', 'cast', 'director', 'keywords', 'genres']].head(3)


# This is our new dataset with just the important parameters

# Finally, I perform some cleaning on the data

# In[ ]:


# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# In[ ]:


# Apply clean_data function to the features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    df2[feature] = df2[feature].apply(clean_data)


# Finally, I created a metadata string fucntion which will help to concat all the important features

# In[ ]:


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
df2['soup'] = df2.apply(create_soup, axis=1)
df2['soup']


# One important thing which I came to know is that here, we need to **CountVectorizer instead of tf-idf** because tf-idf only gives unique values. So if one actor has acted in many films and another actor has acted in only few, then  tf-idf gives same weightage which is tecnically not right. 
# 
# <br> So instead here I use CountVectorizer as it takes into account the number of occurances of a data too

# In[ ]:


# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')

# again we take a part of dataset as our kernel cannot handle all the dataset together.
 
df5 = df2['soup'].head(20000)

count_matrix = count.fit_transform(df5)
 


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# In[ ]:


# we just reset the index of main data
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])


# In[ ]:


#finally calling the get_recommendation fucntion
get_recommendations('Assassins', cosine_sim2)


# ### **The system is now able to recommend movies which are really similar to the one mentioned i.e Assasins. It has taken into account all the important features like the director of the movie, cast, and the various keywords associated with the movie.**

# **Let's try with another movie** 

# In[ ]:


get_recommendations('Jumanji', cosine_sim2)


# **Jumanji being a fantasy movie, the system is able recommend movies which are of the same genre or has similar cast, and keywords**

# ### Summary:
# 
# So, thus I have succesfully been able to develop a recommmender system which basically uses word_to_vec encodings along with tf_idf / CostVectorizer to determine the frequency of words and calculate the matrices
# <br>
# 
# Further I used cosine_similarity score to evaluate the simalarity of words and then accurately recommend the most 10 similar movies to the user based on the title of the movie which the user gives as an input to the system
# 
# ### Difficulties faced: 
# 
# The most difficult part was preprocesing the data. It took me a significant amount of time to understand the data completly, before I could decide what features to include for the recommender system. I had to go through all the columns of the data and had to scrutinize the content, and remove some redundant values as they were becoming an obstacle in merging both the datasets.
# 
# Also since the kaggle kernel gives 16GB of RAM so kernel wasnt able to process the complete data. Nevertheless this has always been an issue with the kaggle kernels but still they are quite handy when it comes to performing some quick ML task :) 
# 
# #### Overall I enjoyed working on this task, referred some tutorials which helped me to clarify my doubts and provided me  more deeper insight into recommendation systems
# 
# 

# In[ ]:




