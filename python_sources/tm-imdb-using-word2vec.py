#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gensim
import nltk
import os
print(os.listdir("../input/embeddings/GoogleNews-vectors-negative300/"))

# Any results you write to the current directory are saved as output.


# In[ ]:


path = "../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"
embeddings = gensim.models.KeyedVectors.load_word2vec_format(path,binary=True)


# In[ ]:


list(embeddings['modi'][:5])


# In[ ]:


pd.Series(embeddings['modi'][:5])


# In[ ]:


embeddings.most_similar('modi',topn=10)


# In[ ]:


url = 'https://bit.ly/2S2yXEd'
data = pd.read_csv(url)
data.head()


# In[ ]:


doc1 = data.iloc[0,0]
print(doc1)
print(nltk.word_tokenize(doc1.lower()))


# In[ ]:


docs = data['review']
docs.head()


# In[ ]:


words = nltk.word_tokenize(doc1.lower())
temp = pd.DataFrame()
for word in words:
    try:
        print(word,embeddings[word][:5])
        temp = temp.append(pd.Series(embeddings[word][:5]),ignore_index=True)
    except:
        print(word,'is not there')


# In[ ]:


temp


# In[ ]:


docs = docs.str.lower().str.replace('[^a-z ]','')
docs.head()


# In[ ]:


from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stopwords  = nltk.corpus.stopwords.words('english')

def clean_doc(doc):
    words = doc.split(' ')
    words_clean = [word for word in words if word not in stopwords]
    doc_clean= ' '.join(words_clean)
    return doc_clean

docs_clean = docs.apply(clean_doc)
docs_clean.head()


# In[ ]:


docs_clean.shape


# In[ ]:


docs_vectors =  pd.DataFrame()

for doc in docs_clean:
    words = nltk.word_tokenize(doc)
    temp =  pd.DataFrame()
    for word in words:
        try:
            word_vec = embeddings[word]
            temp = temp.append(pd.Series(word_vec),ignore_index=True)
        except:
            pass
    docs_vectors=docs_vectors.append(temp.mean(),ignore_index=True)   
docs_vectors.shape    


# In[ ]:


docs_vectors.head()


# In[ ]:


pd.isnull(docs_vectors).sum(axis=1).sort_values(ascending=False).head()


# In[ ]:


X = docs_vectors.drop([64,590])
Y = data['sentiment'].drop([64,590])


# In[ ]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=.2,random_state=100)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score
model = RandomForestClassifier(n_estimators=800)
model.fit(xtrain,ytrain)
test_pred =  model.predict(xtest)
accuracy_score(ytest,test_pred)


# In[ ]:


model = AdaBoostClassifier(n_estimators=800)
model.fit(xtrain,ytrain)
test_pred =  model.predict(xtest)
accuracy_score(ytest,test_pred)


# **HOTSTAR - GO SOLO review Analysis**

# In[ ]:


url = 'https://bit.ly/2W21FY7'
data = pd.read_csv(url)
data.shape


# In[ ]:


data.head()


# In[ ]:


docs = data.loc[:,'Lower_Case_Reviews']
print(docs.shape)
docs.head()


# In[ ]:


Y = data['Sentiment_Manual']
Y.head()


# In[ ]:


Y.value_counts()


# In[ ]:


docs = docs.str.lower().str.replace('[^a-z ]','')
docs.head()


# In[ ]:


from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stopwords  = nltk.corpus.stopwords.words('english')

def clean_doc(doc):
    words = doc.split(' ')
    words_clean = [stemmer.stem(word) for word in words if word not in stopwords]
    doc_clean= ' '.join(words_clean)
    return doc_clean

docs_clean = docs.apply(clean_doc)
docs_clean.head()


# In[ ]:


X = docs_clean 
X.shape,Y.shape


# In[ ]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=.2,random_state=100)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(min_df=5)
cv.fit(X)


# In[ ]:


XTRAIN = cv.transform(xtrain)
XTEST = cv.transform(xtest)


# In[ ]:


XTRAIN = XTRAIN.toarray()
XTEST = XTEST.toarray()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.metrics import accuracy_score
model = dtc(max_depth=10)
model.fit(XTRAIN,ytrain)
yp= model.predict(XTEST)
accuracy_score(ytest,yp)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB as mnb
m1=mnb()
m1.fit(XTRAIN,ytrain)
yp1=m1.predict(XTEST)
accuracy_score(ytest,yp1)


# In[ ]:


from sklearn.naive_bayes import BernoulliNB as bnb
m2=bnb()
m2.fit(XTRAIN,ytrain)
yp2=m2.predict(XTEST)
accuracy_score(ytest,yp2)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(min_df=5)
tv.fit(X)


# In[ ]:


XTRAIN = tv.transform(xtrain)
XTEST = tv.transform(xtest)


# In[ ]:


XTRAIN = XTRAIN.toarray()
XTEST = XTEST.toarray()


# In[ ]:


from sklearn.naive_bayes import MultinomialNB as mnb
mod=mnb()
mod.fit(XTRAIN,ytrain)
ypred=mod.predict(XTEST)
accuracy_score(ytest,ypred)


# In[ ]:


stopwords  = nltk.corpus.stopwords.words('english')

def clean_doc(doc):
    words = doc.split(' ')
    words_clean = [word for word in words if word not in stopwords]
    doc_clean= ' '.join(words_clean)
    return doc_clean

docs_clean = docs.apply(clean_doc)
docs_clean.head()


# In[ ]:


docs_vectors =  pd.DataFrame()

for doc in docs_clean:
    words = nltk.word_tokenize(doc)
    temp =  pd.DataFrame()
    for word in words:
        try:
            word_vec = embeddings[word]
            temp = temp.append(pd.Series(word_vec),ignore_index=True)
        except:
            pass
    docs_vectors=docs_vectors.append(temp.mean(),ignore_index=True)   
docs_vectors.shape    


# In[ ]:


Y.shape


# In[ ]:


df = pd.concat([docs_vectors,Y],axis=1)
df.head(3)


# In[ ]:


df[df.iloc[:,0].isnull()].shape


# In[ ]:


df = df.dropna(axis=0)


# In[ ]:


df.shape


# In[ ]:


X = df.drop(['Sentiment_Manual'],axis=1)
Y = df['Sentiment_Manual']


# In[ ]:


X.shape,Y.shape


# In[ ]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=.2,random_state=100)


# In[ ]:


xtrain.shape,ytrain.shape


# In[ ]:


from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.metrics import accuracy_score
model = dtc(max_depth=10)
model.fit(xtrain,ytrain)
yp= model.predict(xtest)
accuracy_score(ytest,yp)


# In[ ]:


data.head()


# In[ ]:


data.Sentiment_Manual.shape


# In[ ]:


docs_clean.shape


# In[ ]:


from nltk.sentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

def get_sentiment(sentence,analyser=analyser):
    score = analyser.polarity_scores(sentence)['compound']
    if score > 0:
        return 1
    else:
        return 0

