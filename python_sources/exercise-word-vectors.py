#!/usr/bin/env python
# coding: utf-8

# **[Natural Language Processing Home Page](https://www.kaggle.com/learn/natural-language-processing)**
# 
# ---
# 

# # Vectorizing Language
# 
# Embeddings are both conceptually clever and practically effective. 
# 
# So let's try them for the sentiment analysis model you built for the restaurant. Then you can find the most similar review in the data set given some example text. It's a task where you can easily judge for yourself how well the embeddings work.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.nlp.ex3 import *
print("\nSetup complete")


# In[ ]:


# Load the large model to get the vectors
nlp = spacy.load('en_core_web_lg')

review_data = pd.read_csv('../input/nlp-course/yelp_ratings.csv')
review_data.head()


# Here's an example of loading some document vectors. 
# 
# Calculating 44,500 document vectors takes about 20 minutes, so we'll get only the first 100. To save time, we'll load pre-saved document vectors for the hands-on coding exercises.

# In[ ]:


reviews = review_data[:100]
# We just want the vectors so we can turn off other models in the pipeline
with nlp.disable_pipes():
    vectors = np.array([nlp(review.text).vector for idx, review in reviews.iterrows()])
    
vectors.shape


# The result is a matrix of 100 rows and 300 columns. 
# 
# Why 100 rows?
# Because we have 1 row for each column.
# 
# Why 300 columns?
# This is the same length as word vectors. See if you can figure out why document vectors have the same length as word vectors (some knowledge of linear algebra or vector math would be needed to figure this out).

# Go ahead and run the following cell to load in the rest of the document vectors.

# In[ ]:


# Loading all document vectors from file
vectors = np.load('../input/nlp-course/review_vectors.npy')


# # 1) Training a Model on Document Vectors
# 
# Next you'll train a `LinearSVC` model using the document vectors. It runs pretty quick and works well in high dimensional settings like you have here.
# 
# After running the LinearSVC model, you might try experimenting with other types of models to see whether it improves your results.

# In[ ]:


from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(vectors, review_data.sentiment, 
                                                    test_size=0.1, random_state=1)

# Create the LinearSVC model
model = LinearSVC(random_state=1, dual=False)
# Fit the model
____

# Uncomment and run to see model accuracy
# print(f'Model test accuracy: {model.score(X_test, y_test)*100:.3f}%')

# Uncomment to check your work
#q_1.check()


# In[ ]:


# Lines below will give you a hint or solution code
#q_1.hint()
#q_1.solution()


# In[ ]:


# Scratch space in case you want to experiment with other models

#second_model = ____
#second_model.fit(X_train, y_train)
#print(f'Model test accuracy: {second_model.score(X_test, y_test)*100:.3f}%')


# # Document Similarity
# 
# For the same tea house review, find the most similar review in the dataset using cosine similarity.
# 
# # 2) Centering the Vectors
# 
# Sometimes people center document vectors when calculating similarities. That is, they calculate the mean vector from all documents, and they subtract this from each individual document's vector. Why do you think this could help with similarity metrics?
# 
# Run the following line after you've decided your answer.

# In[ ]:


# Check your answer (Run this code cell to receive credit!)
q_2.solution()


# # 3) Find the most similar review
# 
# Given an example review below, find the most similar document within the Yelp dataset using the cosine similarity.

# In[ ]:


review = """I absolutely love this place. The 360 degree glass windows with the 
Yerba buena garden view, tea pots all around and the smell of fresh tea everywhere 
transports you to what feels like a different zen zone within the city. I know 
the price is slightly more compared to the normal American size, however the food 
is very wholesome, the tea selection is incredible and I know service can be hit 
or miss often but it was on point during our most recent visit. Definitely recommend!

I would especially recommend the butternut squash gyoza."""

def cosine_similarity(a, b):
    return np.dot(a, b)/np.sqrt(a.dot(a)*b.dot(b))

review_vec = nlp(review).vector

## Center the document vectors
# Calculate the mean for the document vectors, should have shape (300,)
vec_mean = vectors.mean(axis=0)
# Subtract the mean from the vectors
centered = ____

# Calculate similarities for each document in the dataset
# Make sure to subtract the mean from the review vector
sims = ____

# Get the index for the most similar document
most_similar = ____

# Uncomment to check your work
#q_3.check()


# In[ ]:


# Lines below will give you a hint or solution code
#q_3.hint()
#q_3.solution()


# In[ ]:


print(review_data.iloc[most_similar].text)


# Even though there are many different sorts of businesses in our Yelp dataset, you should have found another tea shop. 
# 
# # 4) Looking at similar reviews
# 
# If you look at other similar reviews, you'll see many coffee shops. Why do you think reviews for coffee are similar to the example review which mentions only tea?

# In[ ]:


# Check your answer (Run this code cell to receive credit!)
q_4.solution()


# # Congratulations!
# 
# You've finished the NLP course. It's an exciting field that will help you make use of vast amounts of data you didn't know how to work with before.
# 
# This course should be just your introduction. Try a project **[with text](https://www.kaggle.com/datasets?tags=14104-text+data)**. You'll have fun with it, and your skills will continue growing.

# ---
# **[Natural Language Processing Home Page](https://www.kaggle.com/learn/natural-language-processing)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum/161466) to chat with other Learners.*
