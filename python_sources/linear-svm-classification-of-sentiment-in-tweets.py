#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Before we begin, we supress deprecation warnings resulting from nltk on Kaggle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# # Linear SVM classification of sentiment in tweets about airlines
# 
# This notebook describes an attempt to classify tweets by sentiment. It describes the initial data exploration, as well as implementation of a linear one-vs-rest Support-Vector-Machine (SVM) classifier.

# ## What is in the dataset?
# 
# It's always good to start by exploring the data that we have available. To do this we load the raw csv file using [Pandas][1] and check what the columns are.
# 
#   [1]: http://pandas.pydata.org/

# In[ ]:


import pandas as pd
tweets = pd.read_csv("../input/Tweets.csv")
list(tweets.columns.values)


# We want to be able to determine the sentiment of a tweet without any other information but the tweet text itself, hence the 'text' column is our focus. Using the text we are going to try and predict 'airline_sentiment'.
# 
# First we take a look at what a typical record looks like.

# In[ ]:


tweets.head()


# Now lets take a look at what sentiments have been found.

# In[ ]:


sentiment_counts = tweets.airline_sentiment.value_counts()
number_of_tweets = tweets.tweet_id.count()
print(sentiment_counts)


# It turns out that our dataset is unbalanced with significantly more negative than positive tweets. We will focus on the issue of identifying negative tweets, and hence treat neutral and positive as one class. It's good to keep in mind that, while a terrible classifier, if we always guessed a tweet was negative we'd be right 62.7% of the time (9178 of 14640). That clearly wouldn't be a very useful classifier, but worth to remember.

# # What characterizes text of different sentiments?
# 
# While we still haven't decided what classification method to use, it's useful to get an idea of how the different texts look. This might be an "old school" approach in the age of deep learning, but lets indulge ourselves nevertheless. 
# 
# To explore the data we apply some crude preprocessing. We will tokenize and lemmatize using [Python NLTK][1], and transform to lower case. As words mostly matter in context we'll look at bi-grams instead of just individual tokens.
# 
# As a way to simplify later inspection of results we will store all processing of data together with it's original form. This means we will extend the Pandas dataframe into which we imported the raw data with new columns as we go along.
# 
# ### Preprocessing
# Note that we remove the first two tokens as they always contain "@ airline_name". We begin by defining our normalization function.
# 
# 
#   [1]: http://www.nltk.org/

# In[ ]:


import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

def normalizer(tweet):
    only_letters = re.sub("[^a-zA-Z]", " ",tweet) 
    tokens = nltk.word_tokenize(only_letters)[2:]
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas


# In[ ]:


normalizer("Here is text about an airline I like.")


# In[ ]:


pd.set_option('display.max_colwidth', -1) # Setting this so we can see the full content of cells
tweets['normalized_tweet'] = tweets.text.apply(normalizer)
tweets[['text','normalized_tweet']].head()


# In[ ]:


from nltk import ngrams
def ngrams(input_list):
    #onegrams = input_list
    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]
    trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]
    return bigrams+trigrams
tweets['grams'] = tweets.normalized_tweet.apply(ngrams)
tweets[['grams']].head()


# And now some counting.

# In[ ]:


import collections
def count_words(input):
    cnt = collections.Counter()
    for row in input:
        for word in row:
            cnt[word] += 1
    return cnt


# In[ ]:


tweets[(tweets.airline_sentiment == 'negative')][['grams']].apply(count_words)['grams'].most_common(20)


# We can already tell there's a pattern here. Sentences like "cancelled flight", "late flight", "booking problems",  "delayed flight" stand out clearly. Lets check the positive tweets.

# In[ ]:


tweets[(tweets.airline_sentiment == 'positive')][['grams']].apply(count_words)['grams'].most_common(20)


# Some more good looking patterns here. We can however see that with 3-grams clear patterns are rare. "great customer service" occurs 12 times in 2362 positive responses, which really doesn't say much in general. 
# 
# Satisfied that our data looks possible to work with begin to construct our first classifier.
# 
# # Linear SVM classifier
# We will build a simple, linear Support-Vector-Machine (SVM) classifier. The classifier will take into account each unique word present in the sentence, as well as all consecutive words. To make this representation useful for our SVM classifier we transform each sentence into a vector. The vector is of the same length as our vocabulary, i.e. the list of all words observed in our training data, with each word representing an entry in the vector. If a particular word is present, that entry in the vector is 1, otherwise 0.
# 
# To create these vectors we use the CountVectorizer from [sklearn][1]. 
# 
# 
#   [1]: http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

# ## Preparing the data

# In[ ]:


import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(ngram_range=(1,2))


# In[ ]:


vectorized_data = count_vectorizer.fit_transform(tweets.text)
indexed_data = hstack((np.array(range(0,vectorized_data.shape[0]))[:,None], vectorized_data))


# In[ ]:


def sentiment2target(sentiment):
    return {
        'negative': 0,
        'neutral': 1,
        'positive' : 2
    }[sentiment]
targets = tweets.airline_sentiment.apply(sentiment2target)


# To check performance of our classifier we want to split our data in to train and test.

# In[ ]:


from sklearn.model_selection import train_test_split
data_train, data_test, targets_train, targets_test = train_test_split(indexed_data, targets, test_size=0.4, random_state=0)
data_train_index = data_train[:,0]
data_train = data_train[:,1:]
data_test_index = data_test[:,0]
data_test = data_test[:,1:]


# ## Fitting the classifier
# 
# We're now ready to fit the classifier to our data. We'll spend more time on hyper parameter tuning later, so for now we just pick some reasonable guesses. Note here that we use the OneVsRestClassifier. This allows us to get the probability distribution over all three classes. Behind the scenes we actually create three classifiers. Each of these classifiers determines the probability that the datapoint belongs to it's corresponding class, or any of the other classes. Hence the name OneVsRest.

# In[ ]:


from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
clf = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear'))
clf_output = clf.fit(data_train, targets_train)


# ## Evaluation of results

# In[ ]:


clf.score(data_test, targets_test)


# It's most likely possible to achieve a higher score with more tuning, or a more advanced approach. Lets check on how it does on a couple of sentences.

# In[ ]:


sentences = count_vectorizer.transform([
    "What a great airline, the trip was a pleasure!",
    "My issue was quickly resolved after calling customer support. Thanks!",
    "What the hell! My flight was cancelled again. This sucks!",
    "Service was awful. I'll never fly with you again.",
    "You fuckers lost my luggage. Never again!",
    "I have mixed feelings about airlines. I don't know what I think.",
    ""
])
clf.predict_proba(sentences)


# So while results aren't very impressive overall, we can see that it's doing a good job on these obvious sentences. 
# 
# ## What is hard for the classifier?
# 
# It's interesting to know which sentences are hard. To find out, lets apply the classifier to all our test sentences and sort by the marginal probability.

# Here are some of the hardest sentences.

# In[ ]:


predictions_on_test_data = clf.predict_proba(data_test)
index = np.transpose(np.array([range(0,len(predictions_on_test_data))]))
indexed_predictions = np.concatenate((predictions_on_test_data, index), axis=1).tolist()


# In[ ]:


def marginal(p):
    top2 = p.argsort()[::-1]
    return abs(p[top2[0]]-p[top2[1]])
margin = sorted(list(map(lambda p : [marginal(np.array(p[0:3])),p[3]], indexed_predictions)), key=lambda p : p[0])
list(map(lambda p : tweets.iloc[data_test_index[int(p[1])].toarray()[0][0]].text, margin[0:10]))


# and their probability distributions?

# In[ ]:


list(map(lambda p : predictions_on_test_data[int(p[1])], margin[0:10]))


# How about the easiest sentences?

# In[ ]:


list(map(lambda p : tweets.iloc[data_test_index[int(p[1])].toarray()[0][0]].text, margin[-10:]))


# and their probability distributions?

# In[ ]:


list(map(lambda p : predictions_on_test_data[int(p[1])], margin[-10:]))


# Looks like all of the easiest sentences are negative. What is the distribution of certainty across all sentences?

# In[ ]:


import matplotlib.pyplot as plt
marginal_probs = list(map(lambda p : p[0], margin))
n, bins, patches = plt.hist(marginal_probs, 25, facecolor='blue', alpha=0.75)
plt.title('Marginal confidence histogram - All data')
plt.ylabel('Count')
plt.xlabel('Marginal confidence')
plt.show()


# Lets break it down by positive and negative sentiment to see if one is harder than the other.
# 
# ### Positive data

# In[ ]:


positive_test_data = list(filter(lambda row : row[0]==2, hstack((targets_test[:,None], data_test)).toarray()))
positive_probs = clf.predict_proba(list(map(lambda r : r[1:], positive_test_data)))
marginal_positive_probs = list(map(lambda p : marginal(p), positive_probs))
n, bins, patches = plt.hist(marginal_positive_probs, 25, facecolor='green', alpha=0.75)
plt.title('Marginal confidence histogram - Positive data')
plt.ylabel('Count')
plt.xlabel('Marginal confidence')
plt.show()


# ### Neutral data

# In[ ]:


positive_test_data = list(filter(lambda row : row[0]==1, hstack((targets_test[:,None], data_test)).toarray()))
positive_probs = clf.predict_proba(list(map(lambda r : r[1:], positive_test_data)))
marginal_positive_probs = list(map(lambda p : marginal(p), positive_probs))
n, bins, patches = plt.hist(marginal_positive_probs, 25, facecolor='blue', alpha=0.75)
plt.title('Marginal confidence histogram - Neutral data')
plt.ylabel('Count')
plt.xlabel('Marginal confidence')
plt.show()


# ### Negative data

# In[ ]:


negative_test_data = list(filter(lambda row : row[0]==0, hstack((targets_test[:,None], data_test)).toarray()))
negative_probs = clf.predict_proba(list(map(lambda r : r[1:], negative_test_data)))
marginal_negative_probs = list(map(lambda p : marginal(p), negative_probs))
n, bins, patches = plt.hist(marginal_negative_probs, 25, facecolor='red', alpha=0.75)
plt.title('Marginal confidence histogram - Negative data')
plt.ylabel('Count')
plt.xlabel('Marginal confidence')
plt.show()


# Clearly the positive data is much harder for the classifier. This makes sense since there's a lot less of it. An important challenge in building a better classifier will then be how to handle positive data.
# 
# A more advanced classifier is described here.
