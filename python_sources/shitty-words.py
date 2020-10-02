#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
import os
print(os.listdir("../input"))


# In[ ]:


train_df=pd.read_csv('../input/train.csv')[:200000]
train_df.shape


# In[ ]:


vectorizer = TfidfVectorizer(min_df=5,strip_accents='unicode',lowercase =True, analyzer='word',use_idf=True, smooth_idf=True, sublinear_tf=True, 
                        stop_words = 'english',tokenizer=word_tokenize)


# # split in words

# In[ ]:


#train_vectorized = vectorizer.transform(train_df.question_text.values)

#train1_tfidf=
vectorizer.fit_transform(train_df[train_df.target==1].question_text.values)
woord1=vectorizer.get_feature_names()
#train0_tfidf=
vectorizer.fit_transform(train_df[train_df.target==0].question_text.values)
woord0=vectorizer.get_feature_names()

woord01=[x for x in set(woord1) if x in set(woord0)]
print(len(woord01),len(woord1),len(woord0))
wordsimpf=[x for x in set(woord1) if x not in set(woord01)]
len(wordsimpf)


# # take glove vh matrix

# In[ ]:


EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]


# In[ ]:



def embedword(word_index,word_pos,word_neg):
    nb_words = min(60000, len(word_index))
    embedding_matrix_1 =pd.DataFrame([])
    i=0
    for word in word_index:
        embedding_vector = embeddings_index.get(word)
        embedding_matrix_1=embedding_matrix_1.append(pd.DataFrame(embedding_vector,columns=[i]).T)
        i=i+1


    #el embeddings_index
    embedding_matrix_1['woord']=word_index
    embedding_matrix_1['target']=0
    for w in word_pos:
        pos=embedding_matrix_1.loc[embedding_matrix_1['woord'] ==w].index
        if pos.size>0:
            embedding_matrix_1.iat[pos[0],301]=1

    for w in word_neg:
        pos=embedding_matrix_1.loc[embedding_matrix_1['woord'] ==w].index
        if pos.size>0:
            embedding_matrix_1.iat[pos[0],301]=2
    embedding_matrix_1.plot.scatter(x=0,y=1,c='target',colormap='viridis')
    return embedding_matrix_1.fillna(0)


# In[ ]:


wh1=embedword(woord1,woord0,wordsimpf)
wh0=embedword(woord0,woord1,wordsimpf)


# In[ ]:


del embeddings_index,EMBEDDING_FILE


# # classify words with the bad words

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier

clf=KNeighborsClassifier(3)
clf = SGDClassifier(max_iter=1000)
clf =XGBClassifier(max_depth=5, base_score=0.005)
wht=wh0.append(wh1)
clf.fit(wht.drop(labels=['woord','target'],axis=1).fillna(0),wht['target'])


# In[ ]:


wht['pred']=clf.predict(wht.drop(labels=['woord','target'],axis=1).fillna(0))  #==wht['target']).mean()


# In[ ]:


wb=wht[wht["pred"]>0].woord

vectorizer2 = TfidfVectorizer(min_df=5,strip_accents='unicode',lowercase =True, analyzer='word',use_idf=True, smooth_idf=True, sublinear_tf=True, 
                        stop_words = 'english',vocabulary=np.unique(wb),tokenizer=word_tokenize)
#vectorizer2 = CountVectorizer(min_df=5,strip_accents='unicode',lowercase =True, analyzer='word',stop_words = 'english',vocabulary=np.unique(wb),tokenizer=word_tokenize)
train_tfidf=vectorizer2.fit_transform([train_df.question_text.sum()] )


# In[ ]:


len( vectorizer2.get_feature_names() )


# # shitty words

# In[ ]:


wordcount=pd.DataFrame(vectorizer2.get_feature_names(),columns=['word'])
wordcount['idf']=1/train_tfidf.T.sum(axis=1)
temp=wht[wht["pred"]>0].groupby("woord").max()
wordcount['target']=temp.target.values
wordcount['pred']=temp.pred.values
wordcount[  wordcount.target>1].sort_values(by=['idf','target','word'])


# In[ ]:


wordcount[  wordcount.pred>1].sort_values(by=['idf','target','word'])

