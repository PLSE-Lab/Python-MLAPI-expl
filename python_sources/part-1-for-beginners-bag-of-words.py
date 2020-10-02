#!/usr/bin/env python
# coding: utf-8

# We could have the origin toturial [word2vec-nlp-tutorial#part-1-for-beginners-bag-of-words][1].  
# Just from the scratch, play with it.
# 
# 
#   [1]: https://www.kaggle.com/c/word2vec-nlp-tutorial#part-1-for-beginners-bag-of-words

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# # Reading the Data  
# The necessary files can be downloaded from the Data page. The first file that you'll need is unlabeledTrainData.tsv, which contains 25,000 IMDB movie reviews, each with a positive or negative sentiment label.

# In[ ]:


# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd       
train = pd.read_csv("../input/labeledTrainData.tsv", header=0,                     delimiter="\t", quoting=3)


# We can make sure that we read 25,000 rows and 3 columns as follows:

# In[ ]:


print("The shape of our data:",train.shape,"\n")

# print columns names
print("Our column names are:",train.columns.values)


# The three columns are called "id", "sentiment", and "array."  Now that you've read the training set, take a look at a few reviews:

# In[ ]:


#print(train["review"][0])


# # Data Cleaning and Text Preprocessing
# Removing HTML Markup: The BeautifulSoup Package
# First, we'll remove the HTML tags. For this purpose, we'll use the [Beautiful Soup][1] library.
# 
# 
#   [1]: https://www.crummy.com/software/BeautifulSoup/bs4/doc/

# In[ ]:


# Import BeautifulSoup into your workspace
from bs4 import BeautifulSoup             

# Initialize the BeautifulSoup object on a single movie review     
example1 = BeautifulSoup(train["review"][0])  

# Print the raw review and then the output of get_text(), for 
# comparison
#print (train["review"][0])
#print (example1.get_text())


# Calling get_text() gives you the text of the review, without tags or markup. If you browse the BeautifulSoup documentation, you'll see that it's a very powerful library - more powerful than we need for this dataset. However, it is not considered a reliable practice to remove markup using regular expressions, so even for an application as simple as this, it's usually best to use a package like BeautifulSoup.
# 
# Dealing with Punctuation, Numbers and Stopwords: NLTK and regular expressions
# When considering how to clean the text, we should think about the data problem we are trying to solve. For many problems, it makes sense to remove punctuation. On the other hand, in this case, we are tackling a sentiment analysis problem, and it is possible that "!!!" or ":-(" could carry sentiment, and should be treated as words. In this tutorial, for simplicity, we remove the punctuation altogether, but it is something you can play with on your own.
# 
# Similarly, in this tutorial we will remove numbers, but there are other ways of dealing with them that make just as much sense. For example, we could treat them as words, or replace them all with a placeholder string such as "NUM".
# 
# To remove punctuation and numbers, we will use a package for dealing with regular expressions, called re. The package comes built-in with Python; no need to install anything. For a detailed description of how regular expressions work, see the [package documentation][1]. Now, try the following:
# 
# 
#   [1]: https://docs.python.org/2/library/re.html#

# In[ ]:


import re
# Use regular expressions to do a find-and-replace
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )  # The text to search
#print (letters_only)


# A full overview of regular expressions is beyond the scope of this tutorial, but for now it is sufficient to know that [] indicates group membership and ^ means "not". In other words, the re.sub() statement above says, "Find anything that is NOT a lowercase letter (a-z) or an upper case letter (A-Z), and replace it with a space."
# 
# We'll also convert our reviews to lower case and split them into individual words (called "tokenization" in NLP lingo):

# In[ ]:


lower_case = letters_only.lower()        # Convert to lower case
words = lower_case.split()               # Split into words


# Finally, we need to decide how to deal with frequently occurring words that don't carry much meaning. Such words are called "stop words"; in English they include words such as "a", "and", "is", and "the". Conveniently, there are Python packages that come with stop word lists built in. Let's import a stop word list from the Python Natural Language Toolkit (NLTK). You'll need to install the library if you don't already have it on your computer; you'll also need to install the data packages that come with it, as follows:

# In[ ]:


from nltk.corpus import stopwords # Import the stop word list
#print (stopwords.words("english") )


# This will allow you to view the list of English-language stop words. To remove stop words from our movie review, do:

# In[ ]:


# Remove stop words from "words"
words = [w for w in words if not w in stopwords.words("english")]
#print (words)


# Don't worry about the "u" before each word; it just indicates that Python is internally representing each word as a unicode string.
# 
# There are many other things we could do to the data - For example, Porter Stemming and Lemmatizing (both available in NLTK) would allow us to treat "messages", "message", and "messaging" as the same word, which could certainly be useful. However, for simplicity, the tutorial will stop here.
# 
# Putting it all together
# Now we have code to clean one review - but we need to clean 25,000 training reviews! To make our code reusable, let's create a function that can be called many times:
# 
# 

# In[ ]:


def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   


# Two elements here are new: First, we converted the stop word list to a different data type, a set. This is for speed; since we'll be calling this function tens of thousands of times, it needs to be fast, and searching sets in Python is much faster than searching lists.
# 
# Second, we joined the words back into one paragraph. This is to make the output easier to use in our Bag of Words, below. After defining the above function, if you call the function for a single review:
# 
# 

# In[ ]:


clean_review = review_to_words( train["review"][0] )
#print (clean_review)


# it should give you exactly the same output as all of the individual steps we did in preceding tutorial sections. Now let's loop through and clean all of the training set at once (this might take a few minutes depending on your computer):

# In[ ]:


# Get the number of reviews based on the dataframe column size
num_reviews = train["review"].size

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in range( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews.append( review_to_words( train["review"][i] ) )


# Sometimes it can be annoying to wait for a lengthy piece of code to run. It can be helpful to write code so that it gives status updates. To have Python print a status update after every 1000 reviews that it processes, try adding a line or two to the code above:

# In[ ]:


print ("Cleaning and parsing the training set movie reviews...\n")
clean_train_reviews = []
for i in range( 0, num_reviews ):
    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        #print ("Review %d of %d\n" % ( i+1, num_reviews ))                                                         
    clean_train_reviews.append( review_to_words( train["review"][i] ))


# # Creating Features from a Bag of Words (Using scikit-learn)
# 
# Now that we have our training reviews tidied up, how do we convert them to some kind of numeric representation for machine learning? One common approach is called a Bag of Words. The Bag of Words model learns a vocabulary from all of the documents, then models each document by counting the number of times each word appears. For example, consider the following two sentences:
# 
# Sentence 1: "The cat sat on the hat"
# 
# Sentence 2: "The dog ate the cat and the hat"
# 
# From these two sentences, our vocabulary is as follows:
# 
# { the, cat, sat, on, hat, dog, ate, and }
# 
# To get our bags of words, we count the number of times each word occurs in each sentence. In Sentence 1, "the" appears twice, and "cat", "sat", "on", and "hat" each appear once, so the feature vector for Sentence 1 is:
# 
# { the, cat, sat, on, hat, dog, ate, and }
# 
# Sentence 1: { 2, 1, 1, 1, 1, 0, 0, 0 }
# 
# Similarly, the features for Sentence 2 are: { 3, 1, 0, 0, 1, 1, 1, 1}
# 
# In the IMDB data, we have a very large number of reviews, which will give us a large vocabulary. To limit the size of the feature vectors, we should choose some maximum vocabulary size. Below, we use the 5000 most frequent words (remembering that stop words have already been removed).
# 
# We'll be using the feature_extraction module from scikit-learn to create bag-of-words features. If you did the Random Forest tutorial in the Titanic competition, you should already have scikit-learn installed; otherwise you will need to install it.

# In[ ]:


print ("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = None,                                max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

# see final the clean data
#print (train_data_features.shape)


# It has 25,000 rows and 5,000 features (one for each vocabulary word).
# 
# Note that CountVectorizer comes with its own options to automatically do preprocessing, tokenization, and stop word removal -- for each of these, instead of specifying "None", we could have used a built-in method or specified our own function to use.  See the function [documentation][1] for more details. However, we wanted to write our own function for data cleaning in this tutorial to show you how it's done step by step.
# 
# Now that the Bag of Words model is trained, let's look at the vocabulary:
# 
# 
#   [1]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

# In[ ]:


# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
#print (vocab)


# If you're interested, you can also print the counts of each word in 
# the vocabulary:

# In[ ]:


import numpy as np

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    # print (count, tag)


# # Random Forest
# 
# At this point, we have numeric training features from the Bag of Words and the original sentiment labels for each feature vector, so let's do some supervised learning! Here, we'll use the Random Forest classifier that we introduced in the Titanic tutorial.  The Random Forest algorithm is included in scikit-learn (Random Forest uses many tree-based classifiers to make predictions, hence the "forest"). Below, we set the number of trees to 100 as a reasonable default value. More trees may (or may not) perform better, but will certainly take longer to run. Likewise, the more features you include for each review, the longer this will take.

# In[ ]:


print( "Training the random forest...")
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["sentiment"] )

