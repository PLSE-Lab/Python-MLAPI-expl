#!/usr/bin/env python
# coding: utf-8

# ### <h1><center>Overview</center></h1>
# 
# ### <center>Sentiment analysis on airline tweets from February 2015 exploring text pre-processing, vectorization, Multinomial Naive Bayes classification. <br> Complementing analysis with data visualizations and feature engineering to get more insight on tweets characteristics. <br><br> Special thanks to Figure Eight for uploading the dataset on kaggle.</center> <br><br>
# 

# In[ ]:


# Libraries to import
import numpy as np 
import pandas as pd
import os
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import re
import string
from nltk.stem.porter import *
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

pd.options.mode.chained_assignment = None

get_ipython().run_line_magic('matplotlib', 'inline')

# Read file
Airline_Tweets = pd.read_csv('../input/Tweets.csv')


# # Exploratory analysis and feature engineering

# In[ ]:


# Check first 5 rows
Airline_Tweets.head()


# In[ ]:


# Count tweets
print(len(Airline_Tweets))


# In[ ]:


# Use lambda expression to check null values for all variables
Airline_Tweets.apply(lambda x: sum(x.isnull()),axis=0)


# In[ ]:


# Print the first 10 tweets and numerate them with enumerate
tweets = list(Airline_Tweets['text'])

for message_no, tweets in enumerate(tweets[:10]):
    print(message_no, tweets)
    print('\n')


# In[ ]:


# Use group by to describe variables by airline sentiment label - could give some indication of what distinguishes each label
Airline_Tweets.groupby('airline_sentiment').mean()


# * If airline_sentiment_confidence and negativereason_confidence refer to likelyhood of the label being correct it seems both variables present good probability average values <br>
# * Low average retweet rate, across all 3 sentiments

# In[ ]:


# Create a column with the lenght of each tweet
Airline_Tweets['length'] = Airline_Tweets['text'].apply(len)


# In[ ]:


# Visualize tweets length
plt.figure(figsize=(10, 7.5))
# Remove the plot frame lines
ax1 = plt.subplot(111)  
ax1.spines["top"].set_visible(False)  
ax1.spines["right"].set_visible(False)  
# Get axis only on the bottom and left of the plot
ax1.get_xaxis().tick_bottom()  
ax1.get_yaxis().tick_left() 
# Format xticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# Format labels
plt.xlabel("Tweets length", fontsize=16)  
plt.ylabel("Count", fontsize=16)
# Plot histogram with Tweets lenght
plt.hist(list(Airline_Tweets['length'].values),  color="darkturquoise", bins=20)
# Change background colour to white
ax1 = plt.gca()
ax1.set_facecolor('w')

plt.show()


# * Distribution is slightly left skewed with mean tweet length (103) < median (114) <br>
# * Spike in tweets around the 140 mark which used to be the maximum length for each tweet at the time (increased to 180 recently)

# In[ ]:


# By running the describe method in tweets length was able to check the longest tweet is 186 characters
# Use conditional selection and iloc to have a look at it
Airline_Tweets[Airline_Tweets['length'] == 186]['text'].iloc[0]


# In[ ]:


# Create a count plot with seaborn to understand how many ocurrences we have for each airline sentiment
plt.figure(figsize=(10, 7.5))
sentiment_ocurrences = sns.countplot(x='airline_sentiment', data=Airline_Tweets, palette='GnBu_d')
# Remove top and right axes
sns.despine()
# Set background colour
sns.set_style(style='white')

sentiment_ocurrences.set_xlabel("Airline sentiment",fontsize=14)
sentiment_ocurrences.set_ylabel("Number of Tweets",fontsize=14)
sentiment_ocurrences.tick_params(labelsize=17)


# * More negative tweets than neutral or positive - something to have in mind for any machine learning tasks

# In[ ]:


# Check if length is a distinguishing feature between a positive, neutral or negative tweet
Airline_Tweets.hist(column='length', by='airline_sentiment', color="darkturquoise", bins=20, figsize=(12,8))

# Change axes and tick labels font size
params = {'axes.titlesize':'12',
          'xtick.labelsize':'12',
          'ytick.labelsize':'12'}
plt.rcParams.update(params)

plt.show()


# * Negative tweets seem to be left skewed <br>
# * Positive and neutral tweets do not seem to show much of a trend. There is a spike around the 140 mark, but they seem to have a more uniform trend

# In[ ]:


# Use datetime to convert tweet_created to datetime
Airline_Tweets['tweet_created'] = pd.to_datetime(Airline_Tweets['tweet_created'])
# Extract hour from tweet_created
Airline_Tweets['tweet_hour'] = Airline_Tweets['tweet_created'].dt.hour
# Define a series of conditions to get part of the day based on the hour
conditions = [
    (Airline_Tweets['tweet_hour'] >= 5) & (Airline_Tweets['tweet_hour'] <= 11),
    (Airline_Tweets['tweet_hour'] >= 12) & (Airline_Tweets['tweet_hour'] <= 18),
    (Airline_Tweets['tweet_hour'] >= 19) & (Airline_Tweets['tweet_hour'] <= 24)]

choices = ['morning', 'afternoon', 'night']
# Create column for part of day
Airline_Tweets['part_of_day'] = np.select(conditions, choices, default='dawn')
# Create iterator to be able to count number of tweets
Airline_Tweets['count'] = 1
# Group part of day and airline sentiment to count all combinations of both
table = Airline_Tweets.groupby(['part_of_day', 'airline_sentiment'])['count'].sum()
# Convert into a dataframe
table = pd.DataFrame(table)
# Reset all indexes
table.reset_index(inplace=True)
# Calculate percentage values
table['Percentage'] = 0
table['Percentage'].loc[table['airline_sentiment'] == 'negative'] = table['count'] / 9178 # 9178 - total number negative tweets
table['Percentage'].loc[table['airline_sentiment'] == 'neutral'] = table['count'] / 3099 # 3099 - total number neutral tweets
table['Percentage'].loc[table['airline_sentiment'] == 'positive'] = table['count'] / 2363 # 2363 - total number positive tweets
table['Percentage'] = round(table['Percentage'] * 100,0) #round percentage
# Specify colour for each group
day_part_colour = {"morning": "powderblue", "afternoon": "c", "night":"c", "dawn":"powderblue"}
# Use catplot from seaborn to plot the results from group by
ax = sns.catplot(x="part_of_day", y="Percentage", col="airline_sentiment",data=table, saturation=.6, kind="bar", ci=None, aspect=.9, palette=day_part_colour, order=['morning', 'afternoon', 'night', 'dawn'])
# Make some adjustments to plot default settings
(ax.set_axis_labels("", "Par of day by sentiment (%)")
  #.set_xticklabels(['morning', 'afternoon', 'night', 'dawn'])
  .set_titles("{col_name} {col_var}")
  .set(ylim=(0, 40))
  .despine(left=True))
# Change the font size
plt.rcParams["axes.labelsize"] = 14
for ax in plt.gcf().axes:
    l = ax.get_xlabel()
    ax.set_xlabel(l, fontsize=14)


# * Most people seem to have tweeted in the afternoon and night across all sentiment states - could related with frequency of flights at those times <br>
# * **Note**: Part of day defined according with following time (debatable as it could change with geographic location):<br>
#     * morning: 5am to 11am<br>
#     * afternoon: 12pm to 6pm<br>
#     * night: 6pm to midnight<br>
#     * dawn: midnight to 5am

# In[ ]:


# Check airline sentiment for each airline
table = Airline_Tweets.groupby(['airline', 'airline_sentiment'])['count'].sum()
# Convert into a dataframe
table = pd.DataFrame(table)
# Reset all indexes
table.reset_index(inplace=True)
# Calculate percentage values
table['Percentage'] = 0
table['Percentage'].loc[table['airline_sentiment'] == 'negative'] = table['count'] / 9178 # 9178 - total number negative tweets
table['Percentage'].loc[table['airline_sentiment'] == 'neutral'] = table['count'] / 3099 # 3099 - total number neutral tweets
table['Percentage'].loc[table['airline_sentiment'] == 'positive'] = table['count'] / 2363 # 2363 - total number positive tweets
table['Percentage'] = round(table['Percentage'] * 100,0) #round percentage
# Use catplot from seaborn to plot the results from group by
ax = sns.catplot(x="airline", y="Percentage", col="airline_sentiment",data=table, saturation=.6, kind="bar", ci=None, aspect=1.5, palette='RdBu')
# Make some adjustments to plot default settings
(ax.set_axis_labels("", "Airline by sentiment (%)")
  .set_titles("{col_name} {col_var}")
  .set(ylim=(0, 40))
  .despine(left=True))
# Change the font size
plt.rcParams["axes.labelsize"] = 14
for ax in plt.gcf().axes:
    l = ax.get_xlabel()
    ax.set_xlabel(l, fontsize=14)


# * Delta, Southwest and Virgin America show good positive and neutral balance when compared to negative sentiment
# * US Airways and United seem to gather most of the negative comments - overall more than 65% of their tweets are negative
# * American and US Airways show bigger gaps between negative and positive or neutral sentiments
# 
# * **Note**: data is slightly inbalanced. Virgin America has only 504 tweets compared to US Airways and United with 2.9k and 3.8k, respectively

# In[ ]:


# Check the most common negative reason
Airline_Tweets['negativereason'].value_counts()


# ## Text pre-processing

# * **Summary of pre-processing steps**: <br><br>
#     * remove tweet handles from the original text column (e.g. @VirginAirlines)
#     * remove special characters using [^a-zA-Z#]
#     * remove short words
#     * remove ponctuation signs using string.punctuation
#     * remove stopwords using stopwords.words('english')
#     * apply stemming by removing all the suffixes from words (e.g. 'cats', 'catlike', and 'catty' all should be the same as the string 'cat')
# 
# * The 5 rows printed after each step will allow one to sense check how the data is changing

# In[ ]:


# Create a new text column that removes tweet handles from text column (e.g. @VirginAirlines)
# Create standard function that will remove twitter handles
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt
# Apply the function above
Airline_Tweets['tidy_tweet'] = np.vectorize(remove_pattern)(Airline_Tweets['text'], "@[\w]*")
# Check first 5 messages
Airline_Tweets['tidy_tweet'].head()


# In[ ]:


# Remove special characters
Airline_Tweets['tidy_tweet'] = Airline_Tweets['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
# Check first 5 messages
Airline_Tweets['tidy_tweet'].head()


# In[ ]:


# Remove words that are very short and might not have any meaning
Airline_Tweets['tidy_tweet'] = Airline_Tweets['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
# Check first 5 messages
Airline_Tweets['tidy_tweet'].head()


# In[ ]:


# To build a classification algorithm one has to transform each tweet text in a numerical feature vector
# Use the bag of words approach to represent each word in the text by a number
def text_process(mess):
    # Make use of string.punctuation to remove pontuation signs
    nopunc = [char for char in mess if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    # Use stopwords.words('english') to remove some of the most common words (e.g. 'the', 'of', 'a')
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

Airline_Tweets['tidy_tweet'] = Airline_Tweets['tidy_tweet'].apply(text_process)
# Check first 5 messages
Airline_Tweets['tidy_tweet'].head()


# In[ ]:


# Create a word cloud at this step to understand what are the most common words used in each tweet
# To speed up processing will split the data between positive and negative sentimet and plot a word cloud each

# Take only positive tweets
positive_tweets = Airline_Tweets.loc[Airline_Tweets['airline_sentiment'] == 'positive']['tidy_tweet']
# Reset the index
positive_tweets.reset_index(inplace=True, drop=True)
# Join each element in a list together
for i in range(len(positive_tweets)):
    positive_tweets[i] = ' '.join(positive_tweets[i])
# Join all the words in a single string
all_positive_words = ' '.join([text for text in positive_tweets])
# Create the wordcloud object
wordcloud = WordCloud(width=600, height=600, background_color="white").generate(all_positive_words)
# Plot graph
plt.figure(figsize=(10, 7.5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)

plt.show()


# In[ ]:


# Take only negative tweets
negative_tweets = Airline_Tweets.loc[Airline_Tweets['airline_sentiment'] == 'negative']['tidy_tweet']
# Reset the index
negative_tweets.reset_index(inplace=True, drop=True)
# Join each element in a list together
for i in range(len(negative_tweets)):
    negative_tweets[i] = ' '.join(negative_tweets[i])
# Join all the words in a single string
all_negative_words = ' '.join([text for text in negative_tweets])
# Create the wordcloud object
wordcloud = WordCloud(width=600, height=600, background_color="white").generate(all_negative_words)
# Plot graph
plt.figure(figsize=(10, 7.5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)

plt.show()


# In[ ]:


# Use PorterStemmer() method to apply stemming to all the tweets
stemmer = PorterStemmer()

Airline_Tweets['tidy_tweet'] = Airline_Tweets['tidy_tweet'].apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
# Check first 5 messages
Airline_Tweets['tidy_tweet'].head()


# In[ ]:


# Put back the column tidy_tweet on this original form with a string per row
clean_tweets = Airline_Tweets['tidy_tweet']

for i in range(len(clean_tweets)):
    clean_tweets[i] = ' '.join(clean_tweets[i])

Airline_Tweets['tidy_tweet'] = clean_tweets

Airline_Tweets['tidy_tweet'].head()


# ## Vectorization
# 
# * **Steps covered in vectorization**: <br><br>
#     * Use CountVectorizer to count the number of times each word occurs in every message following the [bag of words approach](https://en.wikipedia.org/wiki/Bag-of-words_model)
#     * Use TF-IDF to apply weighting and normalization using the following formulas
#         * TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
#         * IDF(t) = log_e(Total number of documents / Number of documents with term t in it)

# In[ ]:


# Use CountVectorizer object to create a matrix will all the words in every tweet
# For this analysis will use the default parameters for CountVectorizer
tweet_transformer = CountVectorizer().fit(Airline_Tweets['tidy_tweet'])

# Print total number of vocabulary words
print(len(tweet_transformer.vocabulary_))


# In[ ]:


# Check an example in detail and take the 4th tweet in the dataset and see its vector representation
# For reference this is the text of the tweet: 'realli aggress blast obnoxi entertain guest face littl recours'
tweet_3 = Airline_Tweets['tidy_tweet'][3]

vector_3 = tweet_transformer.transform([tweet_3])
print(vector_3)
print(vector_3.shape)

# There are 9 unique words in this message and the second number (e.g. 124) will allow one to see what word that is


# In[ ]:


# Apply the transformer in the entire tweets series
tweet_bag_of_words = tweet_transformer.transform(Airline_Tweets['tidy_tweet'])
# Check the shape and number of non-zero ocurrences
print('Shape of Matrix: ', tweet_bag_of_words.shape)
print('Amount of Non-Zero occurences: ', tweet_bag_of_words.nnz)


# In[ ]:


# Adjust the weights with TF-IDF
# Each weight is calculated with the following formula:
# Scenario: in one document with 100 words the word 'data' appears 5 times. There are 1,000 ducoments to classify and the word 'data' appears 90 times in all of them
# TF = 5/100 = 0.05
# IDF = log(1,000/90) = 1
# Tf-idf weight = 0.05 * 1 = 0.05
from sklearn.feature_extraction.text import TfidfTransformer
# Apply the transformer to the bag of words
tweet_tfidf_transformer = TfidfTransformer().fit(tweet_bag_of_words)
tweet_tfidf = tweet_tfidf_transformer.transform(tweet_bag_of_words)
# Check the shape
print(tweet_tfidf.shape)


# ## Build Multinomial Naive Bayes classification model

# * **Objective**: this model will attempt to classify tweets between negative and not negative based on the text of each tweet <br><br>
# * **Model selection**: There is a variety of classification models that could suit this problem. Due to familiarity of approach and ease of implementation will use a Multinomial Naive Bayes classifier that usually [performs well on text classification](https://medium.com/syncedreview/applying-multinomial-naive-bayes-to-nlp-problems-a-practical-explanation-4f5271768ebf) tasks like this one <br><br>
# * **Label simplification**: in line with the objective statement will create a variable **airline_sentiment_model** that will aggregate both positive or neutral sentiments in a single category making it a binary problem: 1 if negative 0 if not negative <br><br>
# * **Potential caveates**: simplifying labels will balance the ratio between negative tweets and not negative, but could come at the risk of losing detail as a postive coment is not exactly the same as a neutral one

# In[ ]:


# Add new column for airline sentiment with binary outcome: 1 for negative comment 0 for not negative
# Create dictionary to map
sentiment_dictionary = {'negative': 1, 'neutral': 0, 'positive': 0}
# Add new column mapping the dictionary
Airline_Tweets['airline_sentiment_model'] = Airline_Tweets['airline_sentiment'].map(sentiment_dictionary)
# Check first 5 rows
Airline_Tweets.head()


# In[ ]:


# Take X and y variables using the TF-IDF vectorization from the previous step
X = tweet_tfidf
y = Airline_Tweets['airline_sentiment_model']
# Do a train test split - will choose to leave the default 30% of the data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Check size of each sample
print(X_train.shape[0], X_test.shape[0], X_train.shape[0] + X_test.shape[0])


# In[ ]:


# Create the Multinomial Naives Bayes object
tweet_sentiment_model = MultinomialNB()
# Fit X_train and y_train to train the model
tweet_sentiment_model.fit(X_train,y_train)


# In[ ]:


# Make one prediction 
print('predicted:', tweet_sentiment_model.predict(X_test)[0])
print('expected:', y_test.iloc[0])


# In[ ]:


# Apply the model to predict X_test values
predictions = tweet_sentiment_model.predict(X_test)


# ## Model evaluation

# In[ ]:


# Print confusion matrix
from sklearn.metrics import confusion_matrix
print (confusion_matrix(y_test, predictions))


# In[ ]:


# Print classification report
from sklearn.metrics import classification_report
print (classification_report(y_test, predictions))


# Performance will always be subjective to what metric is more relevant to the model's objective. <br> Will go through a few key points: <br><br>
# 0 - tweet is not negative <br>
# 1 - tweet is negative <br><br>
# * Model is showing encouraging recall rate, or probability of detection, by classifying correctly 97% of all negative tweets: recall = 2670/(90+2670) = 0.97 <br>
# * Precision indicates there is still potential for improvement with about 74% of negative preditions to be correctly classified: precision = 2670/(2670+924) = 0.74 <br>
# * There are still 924 positive tweets that were incorrectly classified as negative leading to False Positive Rate (FPR) of 55%: FPR = 924/(708+924) = 0.55 <br><br>
# 
# If an airline would be trying to understand why are people tweeting negatively about them, false positives would not be too problematic as this could be filtered out with a deep dive exploratory data analysis. Alternatively, if they would try to follow up on negative tweets by getting into contact with customers then false positives could be an issue, as they could potentially reach out to people that had no problem flying with them.

# # Further topics to explore
# 

# * Understand the difference between using a model only with CountVectorizer() excluding the TF-IDF step <br>
# * Explore hyper parameter tunning in vectorization step and with MultiNomialNB algorithm <br>
# * Test performance with different classification models <br>
# * Extend analysis for longer period and check how seasonality affects negative airline sentiment
