#!/usr/bin/env python
# coding: utf-8

# **This is an attempt to sentiment analysis on the financial tweets.**
# 
# Our goal is :
# 1. Clean the text from the file stockerbot-export1.csv 
# 2. Find the **polarity** for the cleaned text (i.e. **Positive(1)**, **Neutral(0)**, **Negative(-1)**)
# 3. Create **Word Cloud** 
# 4. Creating a **sparse matrix** of all the unique words
# 5. Using **Naive Bayes** for classification and prediction

# In[ ]:


# importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[ ]:


# getting the dataset
dataset = pd.read_csv('../input/sentiment-analysis-on-financial-tweets/stockerbot-export1.csv')


# In[ ]:


dataset.head()


# In[ ]:


dataset = dataset.drop('id',axis=1)


# In[ ]:


dataset.isnull().sum()


# In[ ]:


# rather than removing the null url values, I just replace them with http://www.NULL.com

dataset['url'] = dataset['url'].fillna('http://www.NULL.com')


# **Plotting the top 10 sources with the most financial tweets**

# In[ ]:


plt.figure(figsize=(15,6))
dataset['source'].value_counts()[:10].plot(kind='barh',color=sns.color_palette('summer',30))
plt.title('Source with most number of tweets')


# **Plotting the top 10 url with the most tweets**

# In[ ]:


plt.figure(figsize=(15,6))
dataset['url'].value_counts()[:10].plot(kind='barh',color=sns.color_palette('summer',30))


# **Plotting the top 30 talked about companies in the tweets**

# In[ ]:


plt.figure(figsize=(15,6))
dataset['company_names'].value_counts()[:30].plot(kind='bar',color=sns.color_palette('summer',30))


# ****Cleaning the Tweets****
# 
# We will use the NLTK and re libraries to clean up the text

# In[ ]:


pat1 = r'@[A-Za-z0-9]+' # this is to remove any text with @....
pat2 = r'https?://[A-Za-z0-9./]+'  # this is to remove the urls
combined_pat = r'|'.join((pat1, pat2)) 
pat3 = r'[^a-zA-Z]' # to remove every other character except a-z & A-Z
combined_pat2 = r'|'.join((combined_pat,pat3)) # we combine pat1, pat2 and pat3 to pass it in the cleaning steps


# In[ ]:


len(dataset['text'])


# re.sub() will clean up the text
# 
# tweets.lower() - converting text to lowercase
# 
# tweets.split() - splits the sentence by each word
# 
# ps.stem() - converts the words to lowest degree and we also remove all the stopwords from the text (for example: this, that, etc)
# 
# ' '.join(tweets) - joins back the words to a sentence and separates them with a space

# In[ ]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
cleaned_tweets = []

for i in range(0, len(dataset['text'])) :
    tweets = re.sub(combined_pat2,' ',dataset['text'][i])
    tweets = tweets.lower()
    tweets = tweets.split()
    tweets = [ps.stem(word) for word in tweets if not word in set(stopwords.words('english'))]
    tweets = ' '.join(tweets)
    cleaned_tweets.append(tweets)


# In[ ]:


cleaned_tweets[:10]


# In[ ]:


dataset.columns


# In[ ]:


dataset['cleaned_tweets'] = cleaned_tweets


# **Finding Polarity**
# 
# To find the polarity we use the **SentimentIntensityAnalyzer** from **nltk.sentiment.vader**

# In[ ]:


#nltk.download('vader_lexicon')


# The following code will give us the polarity scores for each of the cleaned tweet 
# 
# For example:
# 
# compound: 0.0, 
#     neg: 0.0, 
#     neu: 1.0, 
#     pos: 0.0, 

# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
for tweet in cleaned_tweets[:10]:
    print(tweet)
    s = sia.polarity_scores(tweet)
    for k in sorted(s):
        print('{0}: {1}, '.format(k, s[k]), end='')
        print()


# Now, based on the 'compound' polarity score and the knowledge of the data, we can choose which tweet falls in the categories of Positive, Negative and Neutral

# In[ ]:


def findpolarity(data):
    sid = SentimentIntensityAnalyzer()
    polarity = sid.polarity_scores(data)
    if(polarity['compound'] >= 0.2):  
        sentiment = 1
    if(polarity['compound'] <= -0.2):
        sentiment = -1 
    if(polarity['compound'] < 0.2 and polarity['compound'] >-0.2):
        sentiment = 0     
    return(sentiment)


# In[ ]:


findpolarity(cleaned_tweets[0])


# In[ ]:


sentiment = []
for i in range(0, len(cleaned_tweets)):
    s = findpolarity(cleaned_tweets[i])
    sentiment.append(s)


# In[ ]:


len(sentiment)


# In[ ]:


len(cleaned_tweets)


# We create a new dataframe to store the cleaned tweets and their respective polarities and save them to a .csv file

# In[ ]:


tweet_sentiment = pd.DataFrame()
tweet_sentiment['cleaned_tweets'] = cleaned_tweets
tweet_sentiment['sentiment'] = sentiment


# In[ ]:


tweet_sentiment.to_csv('tweet_sentiment.csv', index=False)


# In[ ]:


tweet_sentiment.shape[0]


# **Word Cloud**
# 
# To create word clouds of different sentiments, I created three different lists each for positive, negative and neutral tweets

# In[ ]:


positive_tweet = []
negative_tweet = []
neutral_tweet = []

for i in range(0, tweet_sentiment.shape[0]):
    if tweet_sentiment['sentiment'][i] == 0:
        neutral_tweet.append(tweet_sentiment['cleaned_tweets'][i])
    elif tweet_sentiment['sentiment'][i] == 1:
        positive_tweet.append(tweet_sentiment['cleaned_tweets'][i])
    elif tweet_sentiment['sentiment'][i] == -1:
        negative_tweet.append(tweet_sentiment['cleaned_tweets'][i])


# In[ ]:


negative_tweet[:10]


# To create word clouds, we will first have to install word cloud (If using Jupyter Notebook)
# 
# **pip install wordcloud**

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(positive_tweet)
show_wordcloud(neutral_tweet)
show_wordcloud(negative_tweet)


# Now we will use **CountVectorizer** to create a sparse matrix from the cleaned tweets and define the DV and IV for the classification

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(tweet_sentiment['cleaned_tweets']).toarray()
y = tweet_sentiment['sentiment']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)


# For the classification we will use the **Naive Bayes** classifier

# In[ ]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)


# In[ ]:


cm


# In[ ]:


score


# **This is the first version of the analysis and there are a lot of possibilities for improvement. I will try to do the best I can in updating this notebook as frequently as possible. **
# 
# **Any feedback, support and comments will be highly appreciated. **
