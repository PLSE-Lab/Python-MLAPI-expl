#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The goal of this study is to investigate how Twitter, a popular social media platform, can be leveraged in detecting early risk of depression of its users. The study is inspired by [this article](https://time.com/1915/how-twitter-knows-when-youre-depressed/). 

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import re
get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().system('pip install nltk')
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer

get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud

get_ipython().system('pip install tweet-preprocessor')
import preprocessor as p

from gensim.models import KeyedVectors

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# In[ ]:


data_dir = "../input"


# In[ ]:


get_ipython().system('ls {data_dir}')


# ## Classification of depressive and normal tweets

# ### Datasets
# 
# For this analysis, the [Sentiment140](https://www.kaggle.com/kazanova/sentiment140) dataset is used. 

# In[ ]:


encoding = 'ISO-8859-1'
col_names = ['target', 'id', 'date', 'flag', 'user', 'text']

dataset = pd.read_csv(os.path.join(data_dir, 'sentiment140/training.1600000.processed.noemoticon.csv'), encoding=encoding, names=col_names)


# In[ ]:


dataset.head()


# For this experiment, I took a random sample of 8000 tweets.

# In[ ]:


df = dataset.copy().sample(8000, random_state=42)
df["label"] = 0
df = df[['text', 'label']]
df.dropna(inplace=True)
df.head()


# Since there is no readily available public dataset on depression, I found a dataset scraped by [Twint](https://github.com/twintproject/twint).  

# In[ ]:


col_names = ['id', 'text']
df2 = pd.read_csv(os.path.join(data_dir, 'depressive-tweets-processed/depressive_tweets_processed.csv'), sep = '|', header = None, usecols = [0,5], nrows = 3200, names=col_names)


# In[ ]:


df2.info()


# In[ ]:


# add `label` colum with value 1's
df2['label'] = 1
df2 = df2[['text', 'label']]


# In[ ]:


df = pd.concat([df,df2]) # merge the dataset on normal tweets and depressive tweets
df = df.sample(frac=1)  # shuffle the dataset


# In[ ]:


df.info()


# ### Preprocessing

# In[ ]:


contractions = pd.read_json(os.path.join(data_dir, 'english-contractions/contractions.json'), typ='series')
contractions = contractions.to_dict()


# In[ ]:


c_re = re.compile('(%s)' % '|'.join(contractions.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return contractions[match.group(0)]
    return c_re.sub(replace, text)


# In[ ]:


BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

def clean_tweets(tweets):
    cleaned_tweets = []
    for tweet in tweets:
        tweet = str(tweet)
        tweet = tweet.lower()
        tweet = BAD_SYMBOLS_RE.sub(' ', tweet)
        tweet = p.clean(tweet)
        
        #expand contraction
        tweet = expandContractions(tweet)

        #remove punctuation
        tweet = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", tweet).split())

        #stop words
        stop_words = set(stopwords.words('english'))
        word_tokens = nltk.word_tokenize(tweet) 
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        tweet = ' '.join(filtered_sentence)
        
        cleaned_tweets.append(tweet)
        
    return cleaned_tweets


# In[ ]:


X = clean_tweets([tweet for tweet in df['text']])


# ## Word analysis

# In[ ]:


depressive_tweets = [clean_tweets([t for t in df2['text']])]
depressive_words = ' '.join(list(map(str, depressive_tweets)))
depressive_wc = WordCloud(width = 512,height = 512, collocations=False, colormap="Blues").generate(depressive_words)


# In[ ]:


plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(depressive_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# It's easy to spot words that are indicative of depression in these tweets: depression, treatment, suffering, crying, help, struggle, risk, hate, sad, anxiety, disorder, suicide, stress, therapy, mental health, emotional, bipolar.

# ### Tokenization

# In[ ]:


MAX_NUM_WORDS = 10000
tokenizer= Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(X)


# In[ ]:


word_vector = tokenizer.texts_to_sequences(X)


# In[ ]:


word_index = tokenizer.word_index


# In[ ]:


vocab_size = len(word_index)
vocab_size   # num of unique tokens


# In[ ]:


MAX_SEQ_LENGTH = 140
input_tensor = pad_sequences(word_vector, maxlen=MAX_SEQ_LENGTH)


# In[ ]:


input_tensor.shape


# ## Baseline model

# ### TF-IDF classifier

# In[ ]:


corpus = df['text'].values.astype('U')
tfidf = TfidfVectorizer(max_features = MAX_NUM_WORDS) 
tdidf_tensor = tfidf.fit_transform(corpus)


# In[ ]:


tdidf_tensor.shape


# ### Training

# In[ ]:


# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(tdidf_tensor, df['label'].values, test_size=0.3)


# In[ ]:


baseline_model = SVC()
baseline_model.fit(x_train, y_train)


# In[ ]:


predictions = baseline_model.predict(x_test)


# In[ ]:


accuracy_score(y_test, predictions)


# In[ ]:


print(classification_report(y_test, predictions, digits=5))


# ## LTSM model
# 
# Let's improve our model with LTSM. 

# ### Word embedding

# In[ ]:


EMBEDDING_FILE = os.path.join(data_dir, 'googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin.gz')
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)


# In[ ]:


EMBEDDING_DIM = 300
embedding_matrix = np.zeros((MAX_NUM_WORDS, EMBEDDING_DIM))


# In[ ]:


for (word, idx) in word_index.items():
    if word in word2vec.vocab and idx < MAX_NUM_WORDS:
        embedding_matrix[idx] = word2vec.word_vec(word)


# ### Training

# In[ ]:


inp = Input(shape=(MAX_SEQ_LENGTH,))
x = Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(100, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(100, activation="relu")(x)
x = Dropout(0.25)(x)
x = Dense(1, activation="sigmoid")(x)


# In[ ]:


# Compile the model
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(input_tensor, df['label'].values, test_size=0.3)


# In[ ]:


model.fit(x_train, y_train, batch_size=16, epochs=10)


# In[ ]:


preds = model.predict(x_test)


# In[ ]:


preds  = np.round(preds.flatten())
print(classification_report(y_test, preds, digits=5))


# ## Playing with other models

# ### Naive Baye's

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, df.label, test_size=0.3, random_state = 42)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(x_train, y_train)


# In[ ]:


y_pred = nb.predict(x_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, digits=5))


# ## Linear Support Vector

# In[ ]:


from sklearn.linear_model import SGDClassifier

sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(x_train, y_train)


# In[ ]:


y_pred = sgd.predict(x_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, digits=5))


# ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
logreg.fit(x_train, y_train)


# In[ ]:


y_pred = logreg.predict(x_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, digits=5))

