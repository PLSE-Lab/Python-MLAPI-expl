#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align:center;font-size:200%;;">Real or Not? NLP with Disaster Tweets</h1>
# <img src="https://www.kdnuggets.com/wp-content/uploads/slideshare-data-mining-wordle.jpg">

# ## Work path in this notebook, (inspired by [@shahules](https://www.kaggle.com/shahules/tweets-complete-eda-and-basic-modeling) and [@marcovasquez](https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud/notebook)).
# - Basic EDA
# - Data Cleaning
# - Feature engineering 
# - Machine learning models

# * # util Libraries.

# In[ ]:


import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import  Counter
plt.style.use('ggplot')
stop=set(stopwords.words('english'))
import re
from nltk.tokenize import word_tokenize
import gensim
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Embedding,GRU,Dense
from keras.initializers import Constant
import os


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# # Loading the data and print head and get in touch

# In[ ]:


rawtrain = pd.read_csv('../input/nlp-getting-started/train.csv')
rawtest = pd.read_csv('../input/nlp-getting-started/test.csv')
rawtrain.info()


# In[ ]:


#lets convert target to object type.
rawtrain['target'] = rawtrain['target'].astype(object)


# In[ ]:


print('There are {} rows and {} columns in train'.format(rawtrain.shape[0],rawtrain.shape[1]))
print('There are {} rows and {} columns in train'.format(rawtest.shape[0],rawtest.shape[1]))


# In[ ]:


#NaN values
rawtrain.isnull().sum()  


# In[ ]:


# Lets make some world Cloud
text = rawtrain.text.values
wordcloud = WordCloud(width = 5000, height = 2500,background_color = 'white',
                      stopwords = STOPWORDS).generate(str(text))

fig = plt.figure(figsize = (16, 10), facecolor = 'k', edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[ ]:


#Keyword countvalues
rawtrain.keyword.value_counts()[:10].plot.bar(color='green',);


# # Data exploration
# Class distribution

# Before we begin with anything else,let's check the class distribution.There are only two classes 0 and 1.

# In[ ]:


rawtrain.groupby('target').target.value_counts().plot.barh()


# There is a class distribution.There are more tweets with class 0 ( No disaster) than class 1 ( disaster tweets), And sample is not unballanced

# ### Number of characters in tweets

# In[ ]:


fig,(ax1,ax2) = plt.subplots(1,2,figsize = (18,6))

tweet_len = rawtrain[rawtrain['target'] == 1]['text'].str.len()
ax1.hist(tweet_len,color = 'black')
ax1.set_title('disaster tweets')


tweet_len = rawtrain[rawtrain['target'] == 0]['text'].str.len()
ax2.hist(tweet_len,color = 'green')
ax2.set_title('Not disaster tweets')
fig.suptitle('Characters in tweets')
plt.show()


# The distribution of both seems to be almost same.120 t0 140 characters in a tweet are the most common among both.

# ### Length of text in tweets, 

# In[ ]:


fig,(ax1,ax2) = plt.subplots(1,2,figsize = (18,5))
tweet_len = rawtrain[rawtrain['target'] == 1]['text'].str.split().map(lambda x: len(x))
ax1.hist(tweet_len,color = 'orange')
ax1.set_title('disaster tweets')

tweet_len = rawtrain[rawtrain['target'] ==0]['text'].str.split().map(lambda x: len(x))
ax2.hist(tweet_len,color = 'purple')
ax2.set_title('Not disaster tweets')
fig.suptitle('Words in a tweet')
plt.show()


# Sounds we've good distribtion here.

# ###  Average word length in a tweet

# In[ ]:


fig,(ax1,ax2) = plt.subplots(1,2,figsize=(18,6))
word1 = rawtrain[rawtrain['target'] == 1]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word1.map(lambda x: np.mean(x)),ax=ax1,color='red')
ax1.set_title('disaster')

word0 = rawtrain[rawtrain['target']==0]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word0.map(lambda x: np.mean(x)),ax=ax2,color='green')
ax2.set_title('Not disaster')
fig.suptitle('Average word length in each tweet')


# # Feature engineering 
# Based on what we've in the above we can make some interesting feature.

# In[ ]:


data = pd.concat([rawtrain,  rawtest])
data


# ### Feature creation

# In[ ]:


data['body_len'] = data['text'].apply(lambda x: len(x) - x.count(" "))
data.head()


# In[ ]:


import string
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100

data['punct%'] = data['text'].apply(lambda x: count_punct(x))

data.head()


# ## Evaluate created features

# In[ ]:


plt.subplots(1,figsize = (18,8))
bins = np.linspace(0, 200, 40)
plt.hist(data[data['target'] == 1]['body_len'], bins, alpha=0.5, density=True, label='1')
plt.hist(data[data['target'] == 0]['body_len'], bins, alpha=0.5, density=True, label='0')
plt.legend(loc='upper left')
plt.xlim(0,150)
plt.show()


# Target equal to 0 est quite bite longuer than 1. Maybe in urgent situation people do not have time to write lot, or are hurry to post first! But when the text len is more than 60 we have target 1 as longuer!! Intresting.

# In[ ]:


bins = np.linspace(0, 50, 40)
plt.subplots(1,figsize = (18,8))
plt.hist(data[data['target']==1]['punct%'], bins, alpha=0.5, density=True, label='1')
plt.hist(data[data['target']==0]['punct%'], bins, alpha=0.5, density=True, label='0')
plt.legend(loc='upper right')
plt.xlim(0,45)
plt.show()


# We have also more punct, target 0, but notice the peak for 0.

# In[ ]:


data[['punct%', 'body_len']].kurt()


# As we noticed our data are tailled Kurtosis important [ref.](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm), we will deal with this in the next step

# ### Data transformation

# ### Box-Cox Power Transformation
# 
# **Base Form**: $$ y^x $$
# 
# | X    | Base Form           |           Transformation               |
# |------|--------------------------|--------------------------|
# | -2   | $$ y ^ {-2} $$           | $$ \frac{1}{y^2} $$      |
# | -1   | $$ y ^ {-1} $$           | $$ \frac{1}{y} $$        |
# | -0.5 | $$ y ^ {\frac{-1}{2}} $$ | $$ \frac{1}{\sqrt{y}} $$ |
# | 0    | $$ y^{0} $$              | $$ log(y) $$             |
# | 0.5  | $$ y ^ {\frac{1}{2}}  $$ | $$ \sqrt{y} $$           |
# | 1    | $$ y^{1} $$              | $$ y $$                  |
# | 2    | $$ y^{2} $$              | $$ y^2 $$                |
# 
# 
# **Process**
# 1. Determine what range of exponents to test
# 2. Apply each transformation to each value of your chosen feature
# 3. Use some criteria to determine which of the transformations yield the best distribution

# In[ ]:


for i in range(1,7):
    fig = plt.subplots(figsize=(10,4))
    plt.hist((data['punct%'])**(1/i), bins=35)
    plt.hist((data['body_len'])**(1/i), bins=35)
    plt.title(i)
    plt.show()


# - As we go transformation become better and better, I'll chose the third one for punct%. Body_len seem to not respond.
# - We notice also stacked bar in left, that just mean 0, for tweets without punctuations.

# In[ ]:


data['punct%tr'] = data['punct%']**(1/3)


# # Data cleaning

# In[ ]:


def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    links = re.compile(r'https?://\S+|www\.\S+')
    text = links.sub(r'',text)
    tags = re.compile(r'<.*?>')
    text = tags.sub(r'',text)
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text =  emoji_pattern.sub(r'', text)
    tokens = re.split('\W+', text)
    #text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text


# In[ ]:


data['cleaned_text'] = data['text'].apply(lambda x: clean_text(x.lower()))
data.head(20)


# ### Let's stem tweets

# In[ ]:


import nltk
ps = nltk.PorterStemmer()
stopword = nltk.corpus.stopwords.words('english')
def stemming(text):
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopword]
    return text
data['stem_text'] = data['cleaned_text'].apply(lambda x: stemming(x))


# ## Vectorizing data

# In[ ]:



tfidf_vect = TfidfVectorizer(analyzer=stemming)
X_tfidf = tfidf_vect.fit_transform(data['cleaned_text'])
tfidframe = pd.DataFrame(X_tfidf.toarray())

# For concate
data = data.reset_index(drop=True)
X_features = pd.concat([data['body_len'], data['punct%'], tfidframe], axis=1)
X_features.head()


# In[ ]:


### Re organize data for algoritms
train = X_features[:rawtrain.shape[0]]
test = X_features[rawtrain.shape[0]:]
y_train = rawtrain['target'].astype('int')


# In[ ]:


type(y_train.values)


# 
# 
# 
# # Models

# In[ ]:



rf = RandomForestClassifier()
param = {'n_estimators': [10, 50, 100],
        'max_depth': [10, 20,  None]}

gs = GridSearchCV(rf, param, cv=5, n_jobs=-1)
gs_fit = gs.fit(train, y_train)
pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]


# ### To be continued... pleas like if it help, and correct me if I'm wrong or doing un-necessary things.
# - keeplearning
