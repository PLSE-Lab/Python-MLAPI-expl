#!/usr/bin/env python
# coding: utf-8

# # Fun with stemming and feature reduction
# 
# Going off of [Guillaume Payen's](https://www.kaggle.com/gpayen/d/snap/amazon-fine-food-reviews/building-a-prediction-model) 
# illustration of a prediction model, I played with some models that predict the number of stars in a review, 
# rather than whether the review is positive/negative. Specifically, I decided to explore:  
# (i) the effect of feature reduction (a.k.a. dimensionality reduction), and  
# (ii) whether or not stemming your feature data is a good idea for food reviews. 
# 
# The code I used can be found on GitHub at [https://github.com/jonscottstevens/AmazonFineFoods](https://github.com/jonscottstevens/AmazonFineFoods)
# 
# ## To stem or not to stem?
# 
# Stemming takes morphologically complex words and reduces them to their root morphemes.  For example, 
# if the input contains the words "tasty", "tastier", and "tastiest", the suffixes would be stripped off, 
# and all of these forms would be treated as a single stem, "tasti".  My intuition is that such suffixes, 
# especially comparative and superlative suffixes like in "tastier" or "tastiest", could be helpful 
# for predicting, e.g., whether a review is 4 stars or 5 stars.  Indeed, I found that a score prediction model 
# does a bit better when the input data is not stemmed.
# 
# ## Feature dimensionality
# 
# We know because of [Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law) that most words occur only once or twice 
# in a corpus.  Most of these words are probably not too predictive of a review's score, and so we don't 
# need to train our prediction model on every word type (i.e., on every feature).  The sklearn module contains some tools for 
# reducing the number of features in the input data.  I tried the "use k best" approach, where you specify how many features 
# you want (i.e. how many word types to consider as possible predictors of score).  I found that the number of features under consideration
# could be reduced from over 200,000 in the raw input to 10,000 with only modest decrease in predictive power.  The models train much faster this way.
# 
# ## Cleaning and vectorizing the data
# 
# The following code extracts and cleans the input data (including stemming, if you want it), and then 
# vectorizes using tf-idf.

# In[ ]:


import sqlite3
import pandas
import re

# read the data
con = sqlite3.connect('../input/database.sqlite')
dataset = pandas.read_sql_query("SELECT Score, Text FROM Reviews;", con)
texts = dataset['Text']
scores = list(dataset['Score'])

# helper function to strip caps and punctuation
def strip(s):
    return re.sub(r'[^\w\s]','',s).lower()

# this is where the stemming happens (works better without)
#from nltk.stem.porter import PorterStemmer
#stemmer = PorterStemmer()
#print("Stemming...")
#texts = [' '.join([stemmer.stem(w) for w in s.split(' ')]) for s in texts]

# further cleanup
texts = [strip(sentence) for sentence in texts]

# now to vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=1)
features = vectorizer.fit_transform(texts)

# create training and testing data sets, starting with the scores
numReviews = len(scores)
trainingScores = scores[0:int(numReviews*0.8)]
testingScores = scores[int(numReviews*0.8):]


# ## Building the model
# 
# The following function reduces the feature dimensionality using the k best features, and then trains a logistic regression model to predict number of stars for each review in the testing set.  The results are stored in the SQL database. 

# In[ ]:


# addPredictions will reduce feature dimensionality to k,
# & then build a predictive model using those features
def addPredictions(k):
    global con
    c = con.cursor()
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.linear_model import LogisticRegression
    # reduce dimensions
    sel = SelectKBest(chi2,k=k)
    kFeatures = sel.fit_transform(features,scores)
    # split reduced review data into training and testing sets
    trainingFeatures = kFeatures[0:int(numReviews*0.8)]
    testingFeatures = kFeatures[int(numReviews*0.8):]
    # fit the prediction model
    model = LogisticRegression(C=1e5)
    model.fit(trainingFeatures,trainingScores)
    # add the predictions to the database
    try:
        c.execute("alter table Reviews add column k{} integer;".format(str(k)))
    except:
        pass
    for n in xrange(0,len(scores)):
        k = str(k)
        p = str(model.predict(kFeatures[n])[0])
        i = str(n+1)
        c.execute("update Reviews set k{}={} where Id={}".format(k,p,i))
    con.commit()
    


# I tested the following values of k, both on stemmed and unstemmed data.

# In[ ]:


#addPredictions(10)
#addPredictions(100)
#addPredictions(1000)
#addPredictions(10000)
#addPredictions(100000)


# ## Findings
# 
# To evaluate the models, I looked at the average error, i.e. the mean absolute value of the difference between the real score and 
# the predicted score.  The majority of the reviews in the corpus are 5-star reviews, such that always guessing 5 
# provides a good baseline.  The average error for always guessing 5 is about 0.75 (already less than one star off on 
# average).  The prediction models always do better than this, and the error goes down as the number of feature dimensions 
# goes up.  However, the difference in average error levels off at about 10,000 features.  Moreover, we find that 
# the models do better across the board when the data is *not* stemmed.
# 
# The following graph illustrates, where the *red* line is without stemming and the *blue* line is with.
# ![Graph 1](http://www.ling.upenn.edu/~jonsteve/M.png)
# 
# That graph is on a logarithmic scale.  For fun, we can visualize it non-logarithmically
# ![Graph 2](http://www.ling.upenn.edu/~jonsteve/N.png)
# 
# The upshot is that we do a little bit better by leaving our suffixes intact, and that 
# if we need to run models quickly, we can reduce the number of features in the input data, but we 
# probably don't want to go below about 10,000.
