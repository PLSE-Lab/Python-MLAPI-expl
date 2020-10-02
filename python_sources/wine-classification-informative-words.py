#!/usr/bin/env python
# coding: utf-8

# # Wine Classification
# We try to classify wines to "bad" and "good" wines using Naive Bayes Classifier in order to explore the difference between words used to describe "bad" and "good".
# 
# The notebook is also avaiable as a [GitHub repository](https://github.com/DostalJ/WineReviews).

# In[ ]:


import pandas as pd
# from pandas import options
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from matplotlib.style import use
use('ggplot')
# options.mode.chained_assignment = None  # default='warn'


# In[ ]:


df = pd.read_csv('../input/winemag-data_first150k.csv')
df.head(5)


# In[ ]:


# df['description'].apply(func=lambda text: len(word_tokenize(text))).hist()
# plt.title('Number of words used to describe a wine.')
# plt.xlabel('Number of words in description.')
# plt.ylabel('Number of reviews.')
# plt.show()


# ### Data Features
# - *Points*: the number of points WineEnthusiast rated the wine on a scale of 1-100 (though they say they only post reviews for wines that score >=80)
# - *Variety*: the type of grapes used to make the wine (ie Pinot Noir)
# - *Description*: a few sentences from a sommelier describing the wine's taste, smell, look, feel, etc.
# - *Country*: the country that the wine is from
# - *Province*: the province or state that the wine is from
# - *Region 1*: the wine growing area in a province or state (ie Napa)
# - *Region 2*: sometimes there are more specific regions specified within a wine growing area (ie Rutherford inside the Napa Valley), but this value can sometimes be blank
# - *Winery*: the winery that made the wine
# - *Designation*: the vineyard within the winery where the grapes that made the wine are from
# - *Price*: the cost for a bottle of the wine

# ### Explore feature distribution
# We are looking for good labels, i.e. indicators of quality of the wine. We'll explore *Points* and *Price* features.

# In[ ]:


df = df.dropna(subset=['price', 'points'])
df[['price', 'points']].hist(layout=(2,1))
plt.show()


# In[ ]:


plt.scatter(df['points'], df['price'])
plt.xlabel('Points')
plt.ylabel('Price')
plt.show()


# We can see that prize does not add much information. We'll use only *points* as the feature.

# ## Classification
# We'll try to classify the best wines and the worst wines in the dataset. We use 25% of wines with lowest number of points as *"bad"* wines and 25% of wines with the highest number of points as the *"good"* wines.

# In[ ]:


df = df.dropna(subset=['description'])  # drop all NaNs

df_sorted = df.sort_values(by='points', ascending=True)  # sort by points

num_of_wines = df_sorted.shape[0]  # number of wines
worst = df_sorted.head(int(0.25*num_of_wines))  # 25 % of worst wines listed
best = df_sorted.tail(int(0.25*num_of_wines))  # 25 % of best wines listed


# In[ ]:


plt.hist(df['points'], color='grey', label='All')
plt.hist(worst['points'], color='blue', label='Worst')
plt.hist(best['points'], color='red', label='Best')
plt.legend()
plt.show()


# In[ ]:


from nltk.tokenize import word_tokenize
from nltk import FreqDist, NaiveBayesClassifier
from random import shuffle


# First we lower all words in description.

# In[ ]:


worst['words'] = worst['description'].apply(func=lambda text: word_tokenize(text.lower()))
best['words'] = best['description'].apply(func=lambda text: word_tokenize(text.lower()))
worst = worst.dropna(subset=['words'])  # drop all NaNs
best = best.dropna(subset=['words'])  # drop all NaNs


# Make dictionary from 3000 most frequent words.

# In[ ]:


all_words = []  # initialize list of all words
# add all words from 'worst' dataset
for description in worst['words'].values:
    for word in description:
        all_words.append(word)
# add all words from 'best' dataset
for description in best['words'].values:
    for word in description:
        all_words.append(word)
all_words = FreqDist(all_words)  # make FreqList
words_features = list(all_words.keys())[:3000]  # select 3000 most frequent words as words features


# We use 3000 long sparse vector as feature vector (0 if feature word is not presented, 1 if feature word is presented).

# In[ ]:


def find_features(doc):
    """Function for making features out of the text"""
    words = set(doc)  # set of words in description
    features = {}  # feature dictionary
    for w in words_features:  # check if any feature word is presented
        features[w] = bool(w in words)  # write to feature vector
    return features  # return feature vector


# We make features from dataset and randomly shuffle it.

# In[ ]:


featureset = ([(find_features(description), 'worst') for description in worst['words']] +
              [(find_features(description), 'best') for description in best['words']])
shuffle(featureset)  # randomly shuffle dataset


# Now we initialize and train classifier.

# In[ ]:


classifier = NaiveBayesClassifier.train(labeled_featuresets=featureset)


# Print most informative features.

# In[ ]:


classifier.show_most_informative_features(50)

