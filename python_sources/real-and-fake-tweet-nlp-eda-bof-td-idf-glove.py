#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from nltk.corpus import stopwords
from nltk.util import ngrams

from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import classification_report,confusion_matrix

from collections import defaultdict
from collections import Counter
plt.style.use('ggplot')
stop=set(stopwords.words('english'))

import re
from nltk.tokenize import word_tokenize
import gensim
import string

from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM,Dense, SpatialDropout1D, Dropout
from keras.initializers import Constant
from keras.optimizers import Adam


# In[ ]:


train= pd.read_csv('../input/nlp-getting-started/train.csv')
test=pd.read_csv('../input/nlp-getting-started/test.csv')


# In[ ]:


train.head()


# In[ ]:


train.target.unique()


# In[ ]:


train.info()


# In[ ]:


train.describe()


# In[ ]:


c = train['target'].value_counts(ascending=True)
print(c)


# In[ ]:


plt.rcParams['figure.figsize'] = (7, 5)
plt.bar(10,c[1],3, label="Real", color='blue')
plt.bar(15,c[0],3, label="Not", color='green')
plt.legend()
plt.ylabel('Number of examples')
plt.title('Propertion of examples')
plt.show()


# > *Lenght of Non disaster tweets are greater then the real one.. Makes sence people in actual disaster won't have that much time to write longer tweets... *

# In[ ]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
tweet_len=train[train['target']==1]['text'].str.len()
ax1.hist(tweet_len,color='blue')
ax1.set_title('disaster tweets')
tweet_len=train[train['target']==0]['text'].str.len()
ax2.hist(tweet_len,color='green')
ax2.set_title('Not disaster tweets')
fig.suptitle('Characters in tweets')
plt.show()


# In[ ]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
word=train[train['target']==1]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='blue')
ax1.set_title('disaster')
word=train[train['target']==0]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='red')
ax2.set_title('Not disaster')
fig.suptitle('Average word length in each tweet')


# In[ ]:


def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    # replacing the punctuations with no space, 
    # which in effect deletes the punctuation marks 
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)


# In[ ]:


train['text'] = train['text'].apply(remove_punctuation)
train.head(10)


# In[ ]:


def create_corpus(target):
    corpus=[]
    
    for x in train[train['target']==target]['text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus


# In[ ]:


def create_corpus_df(tweet, target):
    corpus=[]
    
    for x in train[train['target']==target]['text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus


# In[ ]:


corpus=create_corpus(0)

dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1
        
top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10]


# In[ ]:


# displaying the stopwords
list(stop)


# In[ ]:


corpus=create_corpus(1)

dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
    


# In[ ]:


np.array(top)


# In[ ]:


counter=Counter(corpus)
most=counter.most_common()
x=[]
y=[]
for word,count in most[:40]:
    if (word not in stop) :
        x.append(word)
        y.append(count)


# In[ ]:


sns.barplot(x=x,y=y)


# In[ ]:


def get_top_tweet_bigrams(corpus,r,n=None):
    vec = CountVectorizer(ngram_range=(r, r)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# In[ ]:


plt.figure(figsize=(20,5))
top_tweet_bigrams=get_top_tweet_bigrams(train['text'],1)[:10]
x,y=map(list,zip(*top_tweet_bigrams))
sns.barplot(x=x,y=y)


# In[ ]:


plt.figure(figsize=(20,5))
top_tweet_bigrams=get_top_tweet_bigrams(train['text'],2)[:10]
x,y=map(list,zip(*top_tweet_bigrams))
sns.barplot(x=x,y=y)


# In[ ]:


plt.figure(figsize=(20,5))
top_tweet_bigrams=get_top_tweet_bigrams(train['text'],3)[:10]
x,y=map(list,zip(*top_tweet_bigrams))
sns.barplot(x=x,y=y)


# In[ ]:


df=pd.concat([train,test])
df.shape


# In[ ]:


def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)


# In[ ]:


df['text']=df['text'].apply(lambda x : remove_URL(x))


# In[ ]:


def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)


# In[ ]:


df['text']=df['text'].apply(lambda x : remove_html(x))


# Fake Tweet

# In[ ]:


corpus_0=create_corpus_df(df,0)
len(corpus_0)


# In[ ]:


corpus_0[:10]


# In[ ]:


plt.figure(figsize=(12,8))
word_cloud = WordCloud(
                          background_color='black',
                          max_font_size = 80
                         ).generate(" ".join(corpus_0[:10]))
plt.imshow(word_cloud)
plt.axis('off')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
word_cloud = WordCloud(
                          background_color='black',
                          max_font_size = 80
                         ).generate(" ".join(corpus_0[:100]))
plt.imshow(word_cloud)
plt.axis('off')
plt.show()


# In[ ]:


corpus_1=create_corpus_df(df,1)
len(corpus_1)


# In[ ]:


corpus_1[:10]


# In[ ]:


plt.figure(figsize=(12,8))
word_cloud = WordCloud(
                          background_color='black',
                          max_font_size = 80
                         ).generate(" ".join(corpus_1[:10]))
plt.imshow(word_cloud)
plt.axis('off')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
word_cloud = WordCloud(
                          background_color='black',
                          max_font_size = 80
                         ).generate(" ".join(corpus_1[:100]))
plt.imshow(word_cloud)
plt.axis('off')
plt.show()


# ****Using Tfidf****
# 
# TF-IDF (term frequency-inverse document frequency) is a statistical measure that evaluates how relevant a word is to a document in a collection of documents. This is done by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents.

# 

# In[ ]:


# TF-IDF scores for all the words in the corpus
def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()

    train = tfidf_vectorizer.fit_transform(data)

    return train, tfidf_vectorizer

X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# In[ ]:


def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):
        lsa = TruncatedSVD(n_components=2)
        lsa.fit(test_data)
        lsa_scores = lsa.transform(test_data)
        color_mapper = {label:idx for idx,label in enumerate(set(test_labels))}
        color_column = [color_mapper[label] for label in test_labels]
        colors = ['black','blue']
        if plot:
            plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))
            orange_patch = mpatches.Patch(color='black', label='Not')
            blue_patch = mpatches.Patch(color='blue', label='Real')
            plt.legend(handles=[orange_patch, blue_patch], prop={'size': 30})

fig = plt.figure(figsize=(16, 16))          
plot_LSA(X_train_tfidf, y_train)
plt.show()


# In[ ]:


def create_corpus(df):
    corpus=[]
    for tweet in tqdm(df['text']):
        words=[word.lower() for word in word_tokenize(tweet)]
        corpus.append(words)
    return corpus


# In[ ]:


corpus=create_corpus(df)


# **Using GloVe  https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove**

# In[ ]:


embedding_dict={}
with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r') as f:
    for line in f:
        values=line.split()
        word = values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
f.close()


# In[ ]:


MAX_LEN=50
tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(corpus)
sequences=tokenizer_obj.texts_to_sequences(corpus)

tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')


# In[ ]:


word_index=tokenizer_obj.word_index
print('Number of unique words:',len(word_index))


# In[ ]:


num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,100))

for word,i in tqdm(word_index.items()):
    if i < num_words:
        emb_vec=embedding_dict.get(word)
        if emb_vec is not None:
            embedding_matrix[i]=emb_vec 


# In[ ]:


model=Sequential()

embedding=Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),
                   input_length=MAX_LEN,trainable=False)

model.add(embedding)
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))


optimzer=Adam(learning_rate=3e-4)

model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


target = train['target'].values
train=tweet_pad[:train.shape[0]]
test=tweet_pad[train.shape[0]:]


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(train,target,test_size=0.2)
print('Shape of train',X_train.shape)
print("Shape of Validation ",X_test.shape)


# In[ ]:


fig = plt.figure(figsize=(16, 16))          
plot_LSA(train,target)
plt.show()


# In[ ]:


history=model.fit(X_train,y_train,batch_size=4,epochs=10,validation_data=(X_test,y_test),verbose=2)


# In[ ]:


train_pred_GloVe = model.predict(test)
#since the train_pred_GloVe value are propability we will round it of and convert it into integrer
train_pred_GloVe_int = train_pred_GloVe.round().astype('int')


# In[ ]:


train_pred_GloVe


# In[ ]:


pred = pd.DataFrame(train_pred_GloVe, columns=['preds'])
pred.plot.hist()


# In[ ]:


submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
submission['target'] = train_pred_GloVe_int
submission.head(10)


# In[ ]:


submission.to_csv("submission.csv", index=False, header=True)


# In[ ]:




