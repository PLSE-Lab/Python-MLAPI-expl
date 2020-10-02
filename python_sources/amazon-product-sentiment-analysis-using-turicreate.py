#!/usr/bin/env python
# coding: utf-8

# # Analyze Product Sentiment

# In[ ]:


get_ipython().system(' pip install turicreate')
import turicreate


# # Read product review data

# In[ ]:


# Reading the data and creating an SFrame of the data
products = turicreate.SFrame.read_csv('../input/reviews-of-amazon-baby-products/amazon_baby.csv')
products


# This is the Amazon Baby Products Dataset and I am going to perform sentiment analysis on it. 

# # Explore data 
# 
# This section includes visualisation and data exploration

# In[ ]:


# Looking at our dataset format
products


# In[ ]:


# Grouping Data by names and number of reviews 
products.groupby('name',operations={'count':turicreate.aggregate.COUNT()}).sort('count',ascending=False)


# # Explore the Vulli Sophie giraffe product

# In[ ]:


giraffe_reviews = products[products['name']=='Vulli Sophie the Giraffe Teether']


# In[ ]:


giraffe_reviews


# In[ ]:


# Number of Vulli Sophie Reviews
len(giraffe_reviews)


# In[ ]:


# Let's look at it in a more categorical format to look at individual ratings 
giraffe_reviews['rating'].show()


# # Building a sentiment classifier

# ## Build word count vectors
# This is the first step in creating a sentiment classifier

# In[ ]:


# With Turicreate, tokenization and vectorization happens with just one function rather than multiple processes
products['word_count'] = turicreate.text_analytics.count_words(products['review'])


# In[ ]:


# Let's look at the dataset to look at the word_count filed which adds the wordcount vector
products


# # Define what is positive and negative sentiment
# 
# It is important to define those because we have a numerical rating array and not a 'thumbs up' and thumbs down'

# In[ ]:


products['rating'].show()


# In[ ]:


# Let's ignore 3 star products because they seem to be neutral in opinion 
products = products[products['rating']!= 3]


# In[ ]:


# Define positive sentiment = 4-star or 5-star reviews
products['sentiment'] = products['rating'] >= 4


# If review is above 4 then you get 1 signifying a positive sentiment and if it's below 4 then it becomes a 0 signifying a negative sentiment. 

# In[ ]:


products.head(20)


# In[ ]:


# Let's look at the distribution of the sentiments across the dataframe
products['sentiment'].show()


# # Let's train our sentiment classifier

# In[ ]:


# Start by splitting the data into training and testing data
train_data,test_data = products.random_split(.8,seed=0)


# In[ ]:


# Building the sentiment model already there in the turiCreate Library
sentiment_model = turicreate.logistic_classifier.create(train_data,
                                                        target='sentiment', 
                                                        features=['word_count'], 
                                                        validation_set=test_data)


# # Evaluate the sentiment model 

# In[ ]:


# Using AUC-ROC curve for evaluation of the model
sentiment_model.evaluate(test_data, metric='roc_curve')


# # Apply the sentiment classifier to better understand the Giraffe reviews

# In[ ]:


giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews, output_type = 'probability')


# In[ ]:


giraffe_reviews


# In[ ]:


giraffe_reviews = products[products['name']== 'Vulli Sophie the Giraffe Teether']


# In[ ]:


giraffe_reviews


# # Sort the Giraffe reviews according to predicted sentiment

# In[ ]:


giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)


# In[ ]:


giraffe_reviews


# In[ ]:


giraffe_reviews.tail()


# ## Show the most positive reviews

# In[ ]:


giraffe_reviews[0]['review']


# In[ ]:


giraffe_reviews[1]['review']


# # Most negative reivews

# In[ ]:


giraffe_reviews[-1]['review']


# In[ ]:


giraffe_reviews[-2]['review']


# In[ ]:




