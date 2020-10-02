#!/usr/bin/env python

#  Author: Angela Chapman
#  Date: 8/6/2014
#
#  This file contains code to accompany the Kaggle tutorial
#  "Deep learning goes to the movies".  The code in this file
#  is for Part 1 of the tutorial on Natural Language Processing.
#
# *************************************** #

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
###(NOW embedded here) from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np

from sklearn import metrics ###(ADDED)
###(below, the embedding of KaggleWord2VecUtility.py)
#!/usr/bin/env python

import re
import nltk

###(here already imported) import pandas as pd
###(here already imported) import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords


###()class KaggleWord2VecUtility(object):
###()    """KaggleWord2VecUtility is a utility class for processing raw HTML text into segments for further learning"""

###()@staticmethod
def review_to_wordlist( review, remove_stopwords=False ):
# Function to convert a document to a sequence of words,
# optionally removing stop words.  Returns a list of words.
#
    # 1. Remove HTML
    review_text = BeautifulSoup(review, "lxml").get_text() ###(added "lxml")
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

# Define a function to split a review into parsed sentences
###()@staticmethod
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
# Function to split a review into parsed sentences. Returns a
# list of sentences, where each sentence is a list of words
#
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences
###(above, the embedding of KaggleWord2VecUtility.py)
   
if __name__ == '__main__':
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), '../input', 'labeledTrainData.tsv'), \
                    header=0, delimiter="\t", quoting=3, skiprows=range(1, 1000))###(omit first 1000 rows)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), '../input', 'labeledTrainData.tsv'), \
                    header=0, delimiter="\t", quoting=3, nrows=1000)###(only first 1000 rows)

    print ('The first review is:')###(every print: made for Python3)
    print (train["review"][0])
    print("\n")###()

    ###()raw_input("Press Enter to continue...")


    ###()print 'Download text data sets. If you already have NLTK datasets downloaded, just close the Python download window...'
    #nltk.download()  # Download text data sets, including stop words

    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []

    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list

    print ("Cleaning and parsing the training set movie reviews...\n")
    for i in range( 0, len(train["review"])):
        clean_train_reviews.append(" ".join(review_to_wordlist(train["review"][i], True)))###(KaggleWord2VecUtility. ..)


    # ****** Create a bag of words from the training set
    #
    print ("Creating the bag of words...\n")


    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    np.asarray(train_data_features)

    # ******* Train a random forest using the bag of words
    #
    print ("Training the random forest (this may take a while)...\n")###()


    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 100)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    forest = forest.fit( train_data_features, train["sentiment"] )



    # Create an empty list and append the clean reviews one by one
    clean_test_reviews = []

    print ("Cleaning and parsing the test set movie reviews...\n")
    for i in range(0,len(test["review"])):
        clean_test_reviews.append(" ".join(review_to_wordlist(test["review"][i], True)))###(KaggleWord2VecUtility. ..)

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    np.asarray(test_data_features)

    # Use the random forest to make sentiment label predictions
    print ("Predicting test labels...\n")
    result = forest.predict(test_data_features)

    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column ###(AND the original sentiment as "sent0")
    output = pd.DataFrame( data={"id":test["id"], "sent0":test["sentiment"], "sentiment":result} )

    # Use pandas to write the comma-separated output file
    ###(omit) output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv'), index=False, quoting=3)
    print(output)###(added)
    ###()print ("Wrote results to Bag_of_Words_model.csv")
    
    ###(follwing lines: added)
    print('Test accuracy is: ')
    print(metrics.accuracy_score(test["sentiment"], result))
    print ("\nPredicting train labels...")
    result0 = forest.predict(train_data_features)
    print('Train accuracy is: ')
    print(metrics.accuracy_score(train["sentiment"], result0))

