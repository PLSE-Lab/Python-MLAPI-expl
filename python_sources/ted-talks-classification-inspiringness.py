#!/usr/bin/env python
# coding: utf-8

# ## Introduction

# The problem is determining whether a TED talk is "inspiring" given its transcript. For our purposes, a talk is inspiring if at least 20% of the ratings are "Inspiring".
# 
# The dataset consist of around 2400 talks.
# 
# I personally found the problem interesting because I've watched a lot of TED talks and I find what they do amazing. Through this, we could determine in advance if a talk would leave the audience inspired. It could also be extended to cater to other reactions - Jaw-dropping, confusing, fascinating, etc. - thereby helping future speakers predertemine the audiences reaction to their talk.

# ## Methodology

# The dataset consist of two `csv` files. There were a lot of columns in the given dataset but for this project the only relevant columns are the transcript and ratings columns. I used bag-of-words to represent the transcript and vectorized it using Tfidf vectorizer to get the term frequencies. The ratings came in a python dictionary format, I converted it to a binary 0 or 1 value - 1 for inspiring and 0 otherwise.
# 
# For preprocessing I used pandas and scikit-learn functions to prepare the data. For postprocessing I used scikit-learn's `confusion_matrix` and `classification_report` functions.
# 
# When the time came to optimize, I found that setting solver to sgd resulted in an increase in accuracy as opposed to using the default adam solver. The accuracy also increased when I increased the max_iter from 10000 to 15000; increasing it further to 20000 resulted in no further increase in accuracy.

# ## Data and Analysis

# There are two csv files; `ted_main.csv` and `transcript.csv`.
# 
# `ted_main.csv` contains 2550 rows and 17 columns:
# - comments
# - description
# - duration
# - event
# - film_date
# - languages
# - main_speaker
# - name
# - num_speaker
# - published_date
# - ratings
# - related_talks
# - speaker_occupation
# - tags
# - title
# - url
# - views
# 
# `transcript.csv` contains 2467 rows and 2 columns:
# - trancript
# - url
# 
# The number of rows don't match as some talks don't have transcripts. Because of this I used `pd.merge(details_df, transcript_df, how='inner', on=['url'])` to have an merged dataframe matching the transcript to it's details via the `url` column.
# 

# ### Preprocessing

# In[ ]:


import pandas as pd


# The data set contained two csv files. ted_main.csv contains details of the talks while transcript.csv contains the transcripts.

# In[ ]:


transcript_df = pd.read_csv('../input/transcripts.csv')
details_df = pd.read_csv('../input/ted_main.csv')


# Get the intersection of the two dataframes on column 'url'

# In[1]:


df = pd.merge(details_df, transcript_df, how='inner', on=['url'])


# Drop irrelevant columns

# In[ ]:


df = df.drop(['main_speaker', 'title', 'views', 'description','comments', 
              'duration', 'event', 'film_date', 'languages', 'num_speaker', 
              'published_date', 'related_talks', 'speaker_occupation', 'tags', 'url'], axis=1)
df


# Get the ratio of Inspiring ratings to the total number of ratings. For our purposes, a talk is inspiring if at least 20% of the ratings are Inspiring

# In[ ]:


import ast
insp_ratios = []
for i in range(0,df.shape[0]):
    test = ast.literal_eval(df.iloc[i]['ratings'])

    rating_count = 0
    inspiring_count = 0
    for rating in test:
        rating_count += rating['count']
        if(rating['name'] == 'Inspiring'):
            inspiring_count = rating['count']
    insp_ratios.append(inspiring_count/rating_count)


# Look at how much of our dataset qualified as 'inspiring'

# In[ ]:


x = 0
for ratio in insp_ratios:
    if ratio >= 0.20:
        x += 1
print(x/len(insp_ratios))


# Convert the ratings in the ratings column to numeric values

# In[ ]:


for i in range(0,df.shape[0]):
    if insp_ratios[i] > 0.2:
        df.at[i, 'ratings'] = 1
    else:
        df.at[i, 'ratings'] = 0


# Get the classifications for our neural network

# In[ ]:


import numpy as np

y = np.asarray(df['ratings'], dtype=np.int64)


# Get the features of the dataset using TfidfVectorizer. TfidfVectorizer computes the frequency of the features as opposed to simply counting them. This is important because not all talks have the same length, long talks would have more word count. If this difference is not accounted for, we may end up with an inaccurate representation of the dataset

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df = 0.05) 

vect = vectorizer.fit(df['transcript'])
transcript_dtm = vectorizer.transform(df['transcript'])


# In[ ]:


x = pd.DataFrame(transcript_dtm.toarray(), columns=vect.get_feature_names())


# Due to an unusual bug, the column 'fit' had to be renamed so that the data could be processed properly.

# In[ ]:


x = x.rename(columns = {'fit': 'fit_feature'})


# In[ ]:


x.shape


# Split the data into training set and test set

# In[ ]:


from sklearn.model_selection import train_test_split

#70% training and 30% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
x_train


# ### Classification

# It took a lot of experimentation with the parameters. I found that setting solver to sgd resulted in an increase in accuracy as opposed to using the default adam solver. The accuracy also increased when I increased the max_iter from 10000 to 15000; increasing it further to 20000 gave no additional increase in accuracy. 

# In[ ]:


from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(100),max_iter=15000,learning_rate_init=0.01, solver='sgd')
mlp.fit(x_train,y_train)


# In[ ]:


predictions = mlp.predict(x_test)


# Print the confusion matrix

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix

print (confusion_matrix(y_test,predictions))


# Print the classification report

# In[ ]:


print(classification_report(y_test,predictions))


# ## Conclusion

# A talk is more than it's word frequencies. Viewer reaction can be influenced by the speaker's looks, body language, voice, even the viewer's mood. Despite the simplicity of the model we reached an accuracy of 80%. 
# 
# Perhaps in the future the analysis could be extended to account for the ordering of the words or even elements beyond the transcript.
# 

# In[ ]:




