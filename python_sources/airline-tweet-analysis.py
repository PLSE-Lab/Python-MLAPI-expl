#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import emoji
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud,STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score,accuracy_score


# In[ ]:


df1 = pd.read_csv('../input/twitter-airline-sentiment/Tweets.csv')
df1.head()


# In[ ]:


sns.countplot(df1.airline_sentiment)


# In[ ]:


sns.countplot(df1.airline,hue = df1.airline_sentiment)


# In[ ]:


df1.info()


# In[ ]:


df1.isna().sum()


# In[ ]:


df2 = df1.drop(['tweet_id', 'airline_sentiment_gold', 'name' , 'negativereason_gold', 'tweet_coord', 'tweet_created', 'tweet_location', 'user_timezone' , 'retweet_count' ] , axis=1)


# In[ ]:


df2['negativereason'] = df2['negativereason'].fillna("Can't Tell")
df2['negativereason_confidence'] = df2['negativereason_confidence'].fillna(0)


# In[ ]:


df2.isna().sum()


# In[ ]:


df2.info()


# In[ ]:


sw = stopwords.words("english")
wl = WordNetLemmatizer()


# In[ ]:


def text_processing(tweet):
    tweet = emoji.demojize(tweet)
    tweet = re.sub(r'@\w+' , ' ', tweet)
    tweet = re.sub(r'http\S+', ' ', tweet)
    tweet = re.sub(r'[^a-zA-Z]' , ' ', tweet)
    tweet = tweet.lower()
    tweet = tweet.split()
    tweet = [wl.lemmatize(word) for word in tweet if not word in sw]
    tweet = ' '.join(tweet)
    return tweet


# In[ ]:


df2['text'] = df2['text'].apply(text_processing)


# In[ ]:


wc = WordCloud(width=800 , height=800, min_font_size=12, background_color='white', stopwords=STOPWORDS).generate(' '.join(df2.text))
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wc)
plt.title("All Tweets Wordcount")
plt.show()


# In[ ]:


cv = CountVectorizer(max_features=2000,ngram_range=(1,3))


# In[ ]:


processed_text = cv.fit_transform(df2['text']).toarray()


# In[ ]:


df3 = pd.concat([df2,pd.DataFrame(processed_text)] , axis=1)


# In[ ]:


df3.negativereason.value_counts()


# In[ ]:


df3.airline.value_counts()


# In[ ]:


df4 = pd.get_dummies(df3['airline'],drop_first=True)
df5 = pd.get_dummies(df3['negativereason'], drop_first=True)


# In[ ]:


df6 = pd.concat([df3.drop(['text','negativereason','airline'], axis=1) , df4, df5] ,axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df6.drop('airline_sentiment',axis=1), df6['airline_sentiment'], test_size=0.30, random_state=4200)


# In[ ]:


reg = LogisticRegression()
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
matrix=confusion_matrix(y_test,predictions)
score=accuracy_score(y_test,predictions)
report=classification_report(y_test,predictions)
print(matrix,score,report)


# In[ ]:


clf =  GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=3, random_state=0)
clf.fit(X_train, y_train)
predictions1 = clf.predict(X_test)
matrix1=confusion_matrix(y_test,predictions1)
score1=accuracy_score(y_test,predictions1)
report1=classification_report(y_test,predictions1)
print(matrix1,score1,report1)


# In[ ]:


rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
predictions2 = rfc.predict(X_test)
matrix2=confusion_matrix(y_test,predictions2)
score2=accuracy_score(y_test,predictions2)
report2=classification_report(y_test,predictions2)
print(matrix2,score2,report2)

