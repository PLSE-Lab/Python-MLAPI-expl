#!/usr/bin/env python
# coding: utf-8

# # Introduction

# Sentiment Analysis is the identification of the sentiment associated with a sentence, phrase or an entire document. Since we are dealing with reviews of drugs here, we would certainly be interested in understanding the sentiment associated with these reviews and try to see which drugs have positive reviews and which ones have negative reviews. 
# 
# Identifying the sentiment of a review can help us make an attempt at predicting the rating that the user would give to the given product

# In[ ]:


# The basic imports
import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))
from tqdm import tqdm


# # The TextBlob library

# We will do this analysis using the open source library called [TextBlob](https://textblob.readthedocs.io/en/dev/quickstart.html), which is a really easy to use Natural Language Processing library written on top of a powerful but relatively harder to use framework called [NLTK](https://www.nltk.org/). TextBlob let's us do our sentiment analysis really fast without having to write a lot of code.
# 
# Let's start by importing TextBlob. It comes pre-loaded in Kaggle kernels. 

# In[ ]:


from textblob import TextBlob


# The first step is to create a TextBlob object. This object is created by passing a simple string. This object is the representation that textblob uses for the given "blob" of text under the hood. It parses the text to let us extract the sentiment or other useful metrics from the sentence.
# 
# Consider a simple dummy review which could be a highly positive one. Note that there are 2 sentences here in the review. 

# In[ ]:


example = "This is a wonderful product. I got amazing results by using it"
blob = TextBlob(example) # create the TextBlob object for this example


# Once the TextBlob object is created, extracting the sentiment is a single command.

# In[ ]:


blob.sentiment


# Using the `sentiment` method spits out 2 numbers - Polarity and Subjectivity. Polarity ranges from -1 to +1. This is a positive review so the polarity is close to 1. Subjectivity ranges from 0 to 1 with 0 for objective/factual sentences and +1 for highly opinionated ones. 
# 
# Let's look at a negative example

# In[ ]:


example = "This is a terrible product. I would not recommend using it"
blob = TextBlob(example)
blob.sentiment


# The sentiment is -1 now, indicating that is a negative review. Note that the subjectivity is still high as it is still an opinion and not a fact

# One can extract the individual components as follows:

# In[ ]:


# Polarity
blob.sentiment.polarity


# In[ ]:


# Subjectivity
blob.sentiment.subjectivity


# Finally let's look at a neutral sentence with no sentiment

# In[ ]:


example = "Jupiter is the largest planet in the Solar System"
blob = TextBlob(example)
blob.sentiment


# The sentence above is a neutral, factual sentence with no sentiment. Hence it has 0 polarity and 0 subjectivity.

# **For our application, polarity is more useful than subjectivity**. We will ignore subjectivity for the time being as we are looking at reviews and they're all fundamentally subjective

# # Let's try it on our data

# In[ ]:


train = pd.read_csv('../input/drugsComTrain_raw.csv')

# Inspect first 5 rows
train.head(5)


# In[ ]:


train.shape


# The data has over 160K rows. Let's extract the review column for the sentiment analysis.

# In[ ]:


reviews = train["review"]
print(reviews[:10]) # First 10


# ## Extract sentiments for all these reviews!

# We'll use tqdm to track our progress. It can take a while to calculate the sentiment for 160K samples

# In[ ]:


sentiments = []
for review in tqdm(reviews):
    blob = TextBlob(review)
    sentiments += [blob.sentiment.polarity]


# ## Add this list of sentiments to the original dataframe

# In[ ]:


train["sentiment"] = sentiments


# Let's randomly sample 10 rows from the data and see what we got

# In[ ]:


train.sample(10)


# Notice that the sentiment values, even for reviews with high ratings tend to be low. This happens because the reviews are **long**. Sentiment analysis works best for short sentences with an obvious sentiment. It is less effective when we have several neutral sentences along with ones with high polarity. The neutral sentences dillute the overall polarity of the text.

# # Correlation with ratings

# Let's see if the newly calculated sentiments have any correlation with the ratings. Our hypothesis would be that reviews with postive reviews would have higher polarity values and vice-versa. 

# In[ ]:


np.corrcoef(train["rating"], train["sentiment"])


# We observe that there is indeed a positive but not very strong correlation between the sentiment of the reviews and the ratings given (0.35). Although the correlation is weak, it is still a useful feature for us to make predictions from as it isn't close to 0. 

# ## Visualizing the correlation

# We notice a gentle upward slope for median sentiment for each of the rating buckets

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(x=np.array(train["rating"]),y=np.array(train["sentiment"]))
plt.xlabel("Rating")
plt.ylabel("Sentiment")
plt.title("Sentiment vs Ratings")
plt.show()


# ## Improving the sentiment analysis

# This was a very simple intro to sentiment analysis. We note that the sentiments calculated are only weakly correlated with the ratings, suggesting that there is a lot of room for improvement. A few basic steps that can be taken to improve our sentiment calculation are:
# 
# 1. Pre-processing: Clean up the text. Do stemming/lemmatization. Remove stop words
# 2. Remove neutral sentences - Do sentiment analysis of individual sentences in a review and remove neutral ones. Then re-run the sentiment analysis on the polar sentences. This will give a more accurate sentiment score
# 
# We would still run into a few road blocks. Some of the harder to solve problems where naive sentiment analysis struggles are:
# 
# 1. Double negations 
# 2. Irony or Sarcasm
# 3. Idioms, pop culture references. 
# 
# Analyzing these would require more advanceed NLP techniques and is an active research field.

# **I hope that you find this kernel useful and informative! Play around with textblob to get an idea of its capabilities and limitations. All the best!**
