#!/usr/bin/env python
# coding: utf-8

# # Implementing a sentiment analysis with a rule based aproach
# 
# This task consists on develop the example developed in class about implementing a sentiment analysis with a rule based approach.

# ### Setting up VADER
# > [VADER](https://github.com/cjhutto/vaderSentiment) (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media.
# 
# First step is to set up *VADER* 

# In[ ]:


# importing required packages
import nltk
from nltk.sentiment import vader


# ### Examples using VADER Sentiment Intensity Analyzer
# Let's see a few examples of sentiment analysis using vader. The scores will show the negative, neutral, positive and compound score. 

# In[ ]:


sia = vader.SentimentIntensityAnalyzer()
sia.polarity_scores("What a terrible restaurant")


# In[ ]:


sia.polarity_scores("terrible")


# Even emoticons are supported

# In[ ]:


sia.polarity_scores(":D")


# In[ ]:


sia.polarity_scores(":/")


# In[ ]:


sia.polarity_scores("the cumin was the kiss of death")


# In[ ]:


sia.polarity_scores("the food was good")


# Exclamations and other punctuations are also supported:

# In[ ]:


sia.polarity_scores("the food was good!")


# In[ ]:


sia.polarity_scores("the food was good!!")


# In[ ]:


sia.polarity_scores("the food was not good!!")


# In[ ]:


# this example doesn't report the expected result
sia.polarity_scores("the food was NOT good!!")


# In[ ]:


sia.polarity_scores("the food was not the worst!!")


# In[ ]:


sia.polarity_scores("the food was GOOD")


# In[ ]:


sia.polarity_scores("the food was so good")


# In[ ]:


sia.polarity_scores("I usually hate seafood but I like this")


# In[ ]:


# Here, the sentence meaning is like the one before, but it is
# being detected as negative, although it should be positive
sia.polarity_scores("I usually hate seafood and I like this")


# ### Use Case: Cornell's Movie Review Data
# For this use case, the following steps are going to be followed: 
# 1. **Download Corpus**. The corpus to use is Cornell dataset of 10,000+ pre-classified movie reviews.
# 2. **Classify with VADER**. checking if it is either positive or negative, using VADER's compound score to decide.
# 3. **Measure accuracy** Calculate the percentage of a accuracy.

# First step is to **read the reviews**. Both positive and negative reviews are going to be read and stored in different python lists.

# In[ ]:


# positive reviews
positiveReviewsFileName = '../input/rt-polarity.pos'
with open(positiveReviewsFileName, encoding='utf-8', errors='ignore') as f:
    positiveReviews = f.readlines()

# first review in the list of positive reviews
positiveReviews[0]


# In[ ]:


# negative reviews
negativeReviewsFileName = '../input/rt-polarity.neg'
with open(negativeReviewsFileName, encoding='utf-8', errors='ignore') as f:
    negativeReviews = f.readlines()
    
# first review in the list of negative reviews
negativeReviews[0]


# The dataset is totally balanced (50% of positive and 50% of negative reviews):

# In[ ]:


# Number of positive and negative reviews
print("Number of positive reviews: " + str(len(positiveReviews)))
print("Number of negative reviews: " + str(len(negativeReviews)))


# Let's import nltk and vader packages

# In[ ]:


import nltk
from nltk.sentiment import vader


# The "compound" polarity is the overall decission, so a **function returning this overall polarity** will be defined:

# In[ ]:


sia = vader.SentimentIntensityAnalyzer()
def vaderSentiment(review):
    """
    Function which returns the compound polarity of a sentence
    """
    return sia.polarity_scores(review)['compound']


# In[ ]:


review = "this is the best restaurant in the city"
vaderSentiment(review)


# As expected, the result of the review is positive.
# 
# Let's now obtain all the scores for the negative labeled reviews. If the sentiment analyzed worked perfect, all those polarity scores would be negative.

# In[ ]:


[vaderSentiment(oneNegativeReview) for oneNegativeReview in negativeReviews]


# There are positive scores, so the accuracy of the analyzer is not perfect.
# 
# Now, a **function which performs the analysis for all the negative and positive samples** is defined. It will return a dictionary with the results of the negative and the positive labeled results.

# In[ ]:


def getReviewSentiments(sentimentCalculator):
    """
    Function which returns a dictionary with the results for 
    positive labeled reviews and negative labeled reviews
    """
    negReviewResult = [sentimentCalculator(oneNegativeReview) for oneNegativeReview in negativeReviews]
    posReviewResult = [sentimentCalculator(onePositiveReview) for onePositiveReview in positiveReviews]
    return {'results-on-positive': posReviewResult, 
           'results-on-negative': negReviewResult}


# Thus, the keys of the dictionary will be 2: the results on positive, and the results on negative.

# In[ ]:


vaderResults = getReviewSentiments(vaderSentiment)
vaderResults.keys()


# In[ ]:


# Number of negative results, should be 5331
len(vaderResults['results-on-negative'])


# In order to **measure accuracies**, the percentage of True Positives and True Negatives, as well as the Overall Accuracy are obtained.

# In[ ]:


positiveReviewsResult = vaderResults['results-on-positive']
pctTruePositive = float(
    sum(x > 0 for x in positiveReviewsResult))/len(positiveReviewsResult)
print("Accuracy on positive reviews = " + "%.2f" % (pctTruePositive*100) + "%")


# In[ ]:


negativeReviewsResult = vaderResults['results-on-negative']
pctTrueNegative = float(
    sum(x > 0 for x in negativeReviewsResult))/len(negativeReviewsResult)
print("Accuracy on negative reviews = " + "%.2f" % (pctTrueNegative*100.0) + "%")


# In[ ]:


totalAccurate = float(sum(x>0 for x in positiveReviewsResult)) + float(sum(x<0 for x in negativeReviewsResult))
total = len(positiveReviewsResult) + len(negativeReviewsResult)
print("Overall accuracy = " + "%.2f" % (totalAccurate*100/total) + "%")


# A function performing the accuracy measurements is defined.

# In[ ]:


def runDiagnostics(vaderResults):
    """
    Accuracy measurements function. It reports (prints) the percentage of True Positives and True Negatives, 
    as well as the Overall Accuracy are obtained, given the results of a vader sentiment analysis
    """
    
    # True positives percentage
    positiveReviewsResult = vaderResults['results-on-positive']
    pctTruePositive = float(
        sum(x > 0 for x in positiveReviewsResult))/len(positiveReviewsResult)
    totalPositives = len(positiveReviewsResult)
    print("Accuracy on positive reviews = " + "%.2f" % (pctTruePositive*100) + "%")

    # True negatives percentage
    negativeReviewsResult = vaderResults['results-on-negative']
    pctTrueNegative = float(
        sum(x > 0 for x in negativeReviewsResult))/len(negativeReviewsResult)
    totalNegatives = len(negativeReviewsResult)
    print("Accuracy on negative reviews = " + "%.2f" % (pctTrueNegative*100.0) + "%")
    
    # Overall accuracy
    totalAccurate = float(sum(x>0 for x in positiveReviewsResult)) + float(sum(x<0 for x in negativeReviewsResult))
    total = totalPositives + totalNegatives
    print("Overall accuracy = " + "%.2f" % (totalAccurate*100.0/total) + "%")


# Let's obtain and discuss the results using the diagnostics function

# In[ ]:


runDiagnostics(vaderResults)


# The obtained results are quite bad, particularly for negative labeled reviews. The percentage obtained is lower than 50%, which means that a random analyzer would even perform better. This happens because it's not deducing the negative context in the reviews as well as it should do. As expected, the overall accuracy is also really bad.
# 
# 
# 
# 

# ### Improving base VADER
# In order to obtain better results using a rule-based approach, some tweaks are going to be done to VADER sentiment analyzer.
# 
# #### WordNet
# > [WordNet](http://wordnet.princeton.edu/) is a large lexical database of English. Nouns, verbs, adjetives and adverbs are grouped into sets of cognitive synonyms (synsets), each expressing a distinct concept. Synsets are interlinked by means of a conceptual-semantic and lexical relations.
# 
# It is availabe in the nltk package, so let's see a few examples to see how it works.

# In[ ]:


from nltk.corpus import sentiwordnet as swn

# List of synsets for 'dog'
list(swn.senti_synsets('dog'))


# In[ ]:


list(swn.senti_synsets('dog'))[0]


# Positive and negative scores of a certain synset of a word can be reported

# The synset in position 3 in the list is "cad", whos main meaning is:
# > "cad": a man who behaves badly or unfairly, esp. toward women.
# 
# Thus, its positive score should be 0, and its negative score should be 1:

# In[ ]:


list(swn.senti_synsets('dog'))[3].pos_score()


# In[ ]:


list(swn.senti_synsets('dog'))[3].neg_score()


# Let's define a function (**superNaiveSentiment**) which will use that feature of WordNet for our Use Case. 
# 
# This function will go through each word of a review and, for its first meaning, it will increment a weight factor if the meaning is positive, and decrease this weight if the meaning is negative. This weight for the word will be added to the overall polarity of the review. The polarity of the whole review will be complete once all the words have been iterated, and the function will return this value.

# In[ ]:


def superNaiveSentiment(review):
    reviewPolarity = 0.0
    numExceptions = 0
    
    for word in review.lower().split():
        weight = 0.0
        try:
            common_meaning = list(swn.senti_synsets(word))[0]
            if common_meaning.pos_score() > common_meaning.neg_score():
                weight += common_meaning.pos_score()
            elif common_meaning.pos_score() < common_meaning.neg_score():
                weight -= common_meaning.neg_score()
        
        except:
            numExceptions += 1
        
        reviewPolarity += weight
    
    return reviewPolarity


# This new defined function replaces the vader sentiment calculator, so it should be fed to the function 'getReviewSentiments' in order to obtain the sentiment analysis results.

# In[ ]:


runDiagnostics(getReviewSentiments(superNaiveSentiment))


# The **results** are not much better than the previous ones. In fact, only the accuracy for negative reviews slightly improved, but it is lower than 50% yet, while positive accuracy and overall accuracy actually are a bit worse.
# 
# Let's try to get better than this including **stopwords**

# In[ ]:


from string import punctuation
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english') + list(punctuation))
list(punctuation)


# In[ ]:


stopwords


# This new function (**naiveSentiment**) is going to be an improvement of the previous one (superNaiveSentiment). It will filter stopwords, and it will also iterate through all the meanings of each word of the review. The number of meanings of each woud will be taken in to account in order to normalize the weight of each word.

# In[ ]:


def naiveSentiment(review):
    reviewPolarity = 0.0
    numExceptions = 0
    
    for word in review.lower().split():
        numMeanings = 0
        if word in stopwords:
            continue
        weight = 0.0
        try:
            for meaning in list(swn.senti_synsets(word)):
                if meaning.pos_score() > meaning.neg_score():
                    weight += (meaning.pos_score() - meaning.neg_score())
                    numMeanings += 1
                elif meaning.pos_score() < meaning.neg_score():
                    weight -= (meaning.neg_score() - meaning.pos_score())
                    numMeanings += 1
        except:
            numExceptions += 1
            
        if numMeanings > 0:
            reviewPolarity += weight/numMeanings
    
    return reviewPolarity


# #### Final Results

# In[ ]:


runDiagnostics(getReviewSentiments(naiveSentiment))


# Now the results shows an improvement. Accuracy on positives is on 75% which is good, and negative accuracy is now at least higher than 50%, so it outperforms a random analyzer. The overall accuracy is almost 60%. 
# 
# Besides the improvement with those tweaks, those results are far from being acceptable. This shows the limitations of a rule-based approach for sentiment analysis problems, specially for historical data and sentences with a bit of complexity in the context, as clearly observed in the bad results for negative reviews.
# 
# In addition, with the approach of the 'naiveSentiment' function, eventhough it gives better results, because of using stopwords, some important words for the analysis like 'not' (would negate a whole sentence) are not taken into account.
# Let's see that limitation with an example:

# In[ ]:


review = 'this is the best restaurant in the city'
naiveSentiment(review)


# In[ ]:


negatedReview = 'this is not the best restaurant in the city'
naiveSentiment(negatedReview)


# In[ ]:


oppositeReview = 'this is the worst restaurant in the city'
naiveSentiment(oppositeReview)


# In[ ]:


negatedOppositeReview = 'this is the worst restaurant in the city'
naiveSentiment(oppositeReview)


# Because of filtering 'not', the result is the same when negating the review, although the negated one should report a negative score.
# 
# Some tweaks in the stopwords -removing significant stopwords for sentiment analysis- would improve the results. However, this shows the difficulty and the load of handwork which must be done in rule-based approaches.

# In[ ]:


stopwords.remove('not')


# In[ ]:


review = 'this is the best restaurant in the city'
naiveSentiment(review)


# In[ ]:


negatedReview = 'this is not the best restaurant in the city'
naiveSentiment(negatedReview)


# Removing 'not' from stopwords is giving the desired result for negated sentences. 
# 
# Let's see how this affects to the use case:

# In[ ]:


runDiagnostics(getReviewSentiments(naiveSentiment))


# It didn't affect much, actually, results slightly got worse...
# 
# So, as stated before, rule-based approaches do not perform good on this type of use cases, and requires a lot of handwork in order to get better results.
