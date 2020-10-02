#!/usr/bin/env python
# coding: utf-8

# # Twitterverse Sentiment Analysis for Spiderman: Into The Spider-Verse(2018)  and Aquaman(2018) Movies
# <img src="https://storage.googleapis.com/kagglesdsdata/datasets/103095/244640/Characters.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1547339029&Signature=B0n%2FnbZldwE0mCj6uc28ylbg7Vn5zQuP5JMmEvWjY6xmp93dEePKPLenXUyEKBrrCuq0rgkuPrL9Q6VztEnePLmcLXwrhrEdnUcC1ONAXlHQWtWjHH8E5bw4MzvZ5qfBTo2I893uhZW2TcEZCukTiqy1hXQ0GyPwDOrWpC9gvuGAZuVrGNyJKdx1m%2FBQKAg2Ois1tksAdQ3%2Fl8S97rzbq6f7u%2BYONS5bhtZCDGB8kIVqDd2OK%2BJguctgTgqW2IbB3IBwsDsDb1ZamYj7cXYZCftU9vGKWLHX97kc03YeK%2FyNRw1Frx8k%2F03aTzRoQLniknZM0Z1aR3X%2BgbLNXj3yAA%3D%3D" width="320" height="320" align="center"/>
# Gathering 1600 tweets with the Spiderman hashtag, and 1600 tweets using #Aquaman, the following graph was produced with the results from a trained Naive Bayes Classifier, a model that classified the sentiments of text as positive or negative. Note that no neutral category was given, so every tweet had to fall in either one of the two categories. Help in getting a thorough categorization will be appreciated.

# In[ ]:


#Get the twitter_samples database with 5000 positive tweets and 5000 negative tweets
import nltk
# nltk.download("twitter_samples")
from nltk.corpus import twitter_samples
print (twitter_samples.fileids())#Should show three files


# In[ ]:


#Clean and Tokenize tweets
import string
import re
 
from nltk.corpus import stopwords 
stopwords_english = stopwords.words('english')
 
from nltk.stem import PorterStemmer
stemmer = PorterStemmer() #This is used to reduce the words to their stem word ('likely' becomes 'like').

from nltk.tokenize import TweetTokenizer

#Use a variable for the positive, negative and all tweets using their respective file name:
pos_tweets = twitter_samples.strings('positive_tweets.json')
neg_tweets = twitter_samples.strings('negative_tweets.json')
all_tweets = twitter_samples.strings('tweets.20150430-223406.json')

#Choose elements to remove:
# Happy Emoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])
# all emoticons (happy + sad)
emoticons = emoticons_happy.union(emoticons_sad)

def cleanAndTokenizeTweets(tweet):
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
 
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
 
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
 
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
 
    tweets_clean = []    
    for word in tweet_tokens:
        if (word not in stopwords_english and # remove stopwords
              word not in emoticons and # remove emoticons
                word not in string.punctuation): # remove punctuation
            #tweets_clean.append(word)
            stem_word = stemmer.stem(word) # stemming word
            tweets_clean.append(stem_word)
 
    return tweets_clean

#Create a bag of words function that cleans and tokenizes tweets, then returns the bag.
def bag_of_words(tweet):
    words = cleanAndTokenizeTweets(tweet)
    words_dictionary = dict([word, True] for word in words)    
    return words_dictionary

#Create a list of bags of words with all positive tweets
pos_tweets_set = []
for tweet in pos_tweets:
    pos_tweets_set.append((bag_of_words(tweet), 'pos')) 
    
#Create a list of bags of words with all negative tweets
neg_tweets_set = []
for tweet in neg_tweets:
    neg_tweets_set.append((bag_of_words(tweet), 'neg'))

# Radomize pos_reviews_set and neg_reviews_set
# doing so will output different accuracy result everytime we run the program
from random import shuffle 
shuffle(pos_tweets_set)
shuffle(neg_tweets_set)
 
test_set = pos_tweets_set[:1000] + neg_tweets_set[:1000]
train_set = pos_tweets_set[1000:] + neg_tweets_set[1000:]
 
print('Test tweets:',len(test_set),'Train tweets:',len(train_set)) # Check the number of tweets in test and train

#Training Classifier and Calculating Accuracy
from nltk import classify
from nltk import NaiveBayesClassifier
 
classifier = NaiveBayesClassifier.train(train_set)
 
accuracy = classify.accuracy(classifier, test_set)

from collections import defaultdict
from nltk.metrics import precision, recall, f_measure, ConfusionMatrix

actual_set = defaultdict(set)
predicted_set = defaultdict(set)
 
actual_set_cm = []
predicted_set_cm = []
 
for index, (feature, actual_label) in enumerate(test_set):
    actual_set[actual_label].add(index)
    actual_set_cm.append(actual_label)
 
    predicted_label = classifier.classify(feature)
 
    predicted_set[predicted_label].add(index)
    predicted_set_cm.append(predicted_label)


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

SpidermanDF = pd.read_csv('../input/movie-tweets/Spiderman Tweets')

#Create bag of words for each of the SpiderVerse tweets
Spider_text_set = []
for text in SpidermanDF['Text']:
    Spider_text_set.append(bag_of_words(text))

#Classify each bag of words
Spider_result = []
for bag in Spider_text_set:
    Spider_result.append(classifier.classify(bag))

SpidermanDF['Sentiment'] = Spider_result

plotSeries = SpidermanDF['Sentiment'].value_counts()

AquamanDF = pd.read_csv('../input/movie-tweets/Aquaman Tweets')

Aquaman_text_set = []
for text in AquamanDF['Text']:
    Aquaman_text_set.append(bag_of_words(text))

#Classify each bag of words
Aquaman_result = []
for bag in Aquaman_text_set:
    Aquaman_result.append(classifier.classify(bag))

AquamanDF['Sentiment'] = Aquaman_result #Create a column with the category given
plotSeries1 = AquamanDF['Sentiment'].value_counts()


fig = plt.figure(facecolor="white",figsize=(10,8))
bar_width = 0.4
ax = fig.add_subplot(1, 1, 1)
r = [0,0.5] #Space between bars
tick_pos = [i + (bar_width/40) for i in r]

ax1 = ax.bar(r, plotSeries.values, width=bar_width, label='Spiderman', color='#B11313',edgecolor='white' )
ax2 = ax.bar(r, plotSeries1.values, bottom=plotSeries.values, width=bar_width, label='Aquaman', 
             color='#006994',edgecolor='white')
ax.set_ylabel("Count", fontsize=14, style='italic')
ax.set_xlabel("Sentiment", fontsize=14, style='italic')
ax.legend(loc='best')
plt.xticks(tick_pos, ["Positive", "Negative"], fontsize=14)
plt.yticks(fontsize=13)

for r1, r2 in zip(ax1, ax2): #Code to configure text inside each plot
    h1 = r1.get_height()
    h2 = r2.get_height()
    plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., "%d" % h1, ha="center", va="center", 
             color="white", fontsize=13, fontweight="bold")
    plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., "%d" % h2, ha="center", va="center", 
             color="white", fontsize=13, fontweight="bold")
plt.legend(prop={'size': 13})
plt.title('Sentiment Analysis on Tweets about\n Spiderman and Aquaman Movies')
plt.show()


# ## Box Office Results (as of Jan 8 2019):
# How popular are these movies in terms of ticket sales? The total US box office grosses for each movie are given below:
# 
# __*<font color=#006994>Aquaman(2018) :  USD 266,460,074</font>*__
# 
# __*<font color=#B11313>Spiderman: Into The Spider-Verse(2018) :  USD 136,847,588</font>*__

# ## Classifier Precision Stats
# pos = positive
# 
# neg = negative

# In[ ]:


print('pos precision:', precision(actual_set['pos'], predicted_set['pos']))
print('pos recall:', recall(actual_set['pos'], predicted_set['pos'])) 
print('pos F-measure:', f_measure(actual_set['pos'], predicted_set['pos']))
print('neg precision:', precision(actual_set['neg'], predicted_set['neg']))
print('neg recall:', recall(actual_set['neg'], predicted_set['neg'])) 
print('neg F-measure:', f_measure(actual_set['neg'], predicted_set['neg'])) 


# ## Classifier Acurracy on Labeling Sentiment
# The following matrix shows that the classifier used in this case may have incorrectly labeled a percentage of tweets as negative/positive:

# In[ ]:


cm = ConfusionMatrix(actual_set_cm, predicted_set_cm)
print (cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))


# *Acknowledgements: Mukesh Chapagain in his blog found [here.](http://blog.chapagain.com.np/python-nltk-twitter-sentiment-analysis-natural-language-processing-nlp/)*
# 
# *Spiderman doll image courtesy of amazon.com and Aquaman figure image courtesy of entertainmentearth.com*

# In[ ]:




