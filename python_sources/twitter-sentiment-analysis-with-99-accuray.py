#!/usr/bin/env python
# coding: utf-8

# __What is Sentiment Analysis?__
# 
# Sentiment analysis is a process of identifying an attitude of the author on a topic that is being written about. 

# #### Table of contents
# 
# 1. Load the libraries
# 1. Load Dataset
# 1. EDA
# 1. Preprocessing Tweet Text
# 1. Featurization
#     1. Bag-of-Words
#     1. TF-IDF
#     1. Word2vec
# 1. Resample
#     1. Upsampling BOW
#     2. Upsampling TF-IDF
#     1. Upsampling word2vec
# 1. Split Dataset
# 1. Model Selection
#     1. KNN
# 1. Summary
# 

# # Load the libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem import PorterStemmer,WordNetLemmatizer
from wordcloud import WordCloud
import re
import gensim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.utils import resample
from sklearn.metrics import accuracy_score,f1_score


# # Load Dataset

# In[ ]:


df=pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv')
test=pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/test.csv')
df.head(4)


# #  EDA

# In[ ]:


df.info()


# In[ ]:


print('shape of train dataset',df.shape)
df.label.value_counts()


# In[ ]:


# 
sns.countplot(df.label,)
plt.xlabel('class label')
plt.ylabel('number of tweets')
plt.show()


# In[ ]:


plt.hist(df[df['label']==1].tweet.str.len(),bins=20,label='class 1')
plt.legend()
plt.xlabel('length of tweets')
plt.ylabel('number of tweets')
plt.show()
plt.hist(df[df['label']==0].tweet.str.len(),color='yellow',bins=20,label='class 0')
plt.legend()
plt.xlabel('length of tweets')
plt.ylabel('number of tweets')
plt.show()


# #   Preprocessing Tweet Text
# 
# 1. Removing Twitter Handles (@user)
# 2. Removing urls from text 
# 3. Removing Punctuations, Numbers, and Special Characters
# 
# 5. Convert the word to lowercase
# 6. Remove Stopwords
# 7. Stemming the word
# 8. Lemmatization<br>
# 
# After which we collect the words used to describe positive and negative reviews

# In[ ]:


text=df['tweet'].values.tolist()
text_test=test['tweet'].values.tolist()
text+=text_test
print(len(text))


# In[ ]:


import nltk
stopword=nltk.corpus.stopwords.words('english')
stopword.remove('not')
for index,text_ in enumerate(text):
    text_=re.sub(r'@[\w]*','',text_) #Removing Twitter Handles (@user)
    text_=re.sub(r'http/S+','',text_) #Removing urls from text 
    text_=re.sub(r'[^A-Za-z#]',' ',text_) #Removing Punctuations, Numbers, and Special Characters
    text_=" ".join(i.lower() for i in text_.split() if i.lower() not in stopword) #Removing stopword
    text[index]=text_


# In[ ]:


#Stemming the word
pt=PorterStemmer()
wordnet=WordNetLemmatizer()
for index,text_ in enumerate(text):
    text_=" ".join(pt.stem(i) for i in text_.split())
    text_=" ".join(wordnet.lemmatize(i) for i in text_.split())  
    text[index]=text_


# In[ ]:


df['preprocess_tweet']=text[:len(df)]
df['length_tweet']=df['preprocess_tweet'].str.len()
test['preprocess_tweet']=text[len(df):]
df.head()


# In[ ]:





# In[ ]:





# # Featurization

# ### BOW

# In[ ]:


train=df.copy()
train.drop(columns=['id','tweet','preprocess_tweet'],inplace=True)


# In[ ]:


bow=CountVectorizer( min_df=2, max_features=1000)
bow.fit(df['preprocess_tweet'])
bow_df=bow.transform(df['preprocess_tweet']).toarray()
print('feature name==',bow.get_feature_names()[:10])
print('number of uniqe words',bow_df.shape[1])
print('shape',bow_df.shape)
bow_train=pd.DataFrame(bow_df)
bow_train['length_tweet']=df['length_tweet']
bow_train['label']=df['label']
bow_train.head()


# ### TF-IDF Features (Bi-Grams)

# In[ ]:


tfidf=TfidfVectorizer(ngram_range=(1, 2),min_df=2,max_features=1000)
tfidf.fit(df['preprocess_tweet'])
tfidf_df=tfidf.transform(df['preprocess_tweet']).toarray()
print('number of uniqe words',bow_df.shape[1])
print('shape',tfidf_df.shape)
tfidf_train=pd.DataFrame(tfidf_df)
tfidf_train['length_tweet']=df['length_tweet']
tfidf_train['label']=df['label']
tfidf_train.head()


# ### Word2vec
# __size:__ The number of dimensions of the embeddings and the default is 100.
# 
# __window:__ The maximum distance between a target word and words around the target word. The default window is 5.
# 
# __min_count:__ The minimum count of words to consider when training the model; words with occurrence less than this count will be ignored. The default for min_count is 5.
# 
# __workers:__ The number of partitions during training and the default workers is 3.
# 
# __sg:__ The training algorithm, either CBOW(0) or skip gram(1). The default training algorithm is CBOW.

# In[ ]:


tokenize=df['preprocess_tweet'].apply(lambda x: x.split())
w2vec_model=gensim.models.Word2Vec(tokenize,min_count = 1, size = 100, window = 5, sg = 1)
w2vec_model.train(tokenize,total_examples= len(df['preprocess_tweet']),epochs=20)


# In[ ]:


w2vec_model.most_similar('father')


# In[ ]:


w2v_words = list(w2vec_model.wv.vocab)
print("number of words that occured minimum 5 times ",len(w2v_words))
print("sample words ", w2v_words[0:50])


# In[ ]:


vector=[]
from tqdm import tqdm
for sent in tqdm(tokenize):
  sent_vec=np.zeros(100)
  count =0
  for word in sent: 
    if word in w2v_words:
      vec = w2vec_model.wv[word]
      sent_vec += vec 
      count += 1
  if count != 0:
    sent_vec /= count #normalize
  vector.append(sent_vec)
print(len(vector))
print(len(vector[0]))        


# In[ ]:


#example
l='father dysfunct selfish drag kid dysfunct'
count=0
vcc=np.zeros(100)
for word in l:
  if word in w2v_words:
    v=w2vec_model.wv[word]
    vcc+=v
    count+=1
vcc


# In[ ]:



print('number of uniqe words',len(vector[1]))
w2v_train=pd.DataFrame(vector)
w2v_train['length_tweet']=df['length_tweet']
w2v_train['label']=df['label']
w2v_train.head()


# In[ ]:





# # Resample

# ## Upsampling BOW

# In[ ]:


major_class_0,major_class_1=bow_train.label.value_counts()
df_major=bow_train[bow_train['label']==0]
df_minor=bow_train[bow_train['label']==1]
df_minor_upsampled = resample(df_minor, 
                                 replace=True,     # sample with replacement
                                 n_samples=major_class_0)
df_bow_upsampled = pd.concat([df_major, df_minor_upsampled])
print('shape',df_bow_upsampled.shape)
sns.countplot(df_bow_upsampled.label)


# ## Upsampling TF-IDF

# In[ ]:


major_class_0,major_class_1=tfidf_train.label.value_counts()
df_major=tfidf_train[tfidf_train['label']==0]
df_minor=tfidf_train[tfidf_train['label']==1]
df_minor_upsampled = resample(df_minor, 
                                 replace=True,     # sample with replacement
                                 n_samples=major_class_0)
df_tfidf_upsampled = pd.concat([df_major, df_minor_upsampled])
print('shape',df_tfidf_upsampled.shape)
sns.countplot(df_tfidf_upsampled.label)


# ## Upsampling  word2vec

# In[ ]:


major_class_0,major_class_1=w2v_train.label.value_counts()
df_major=w2v_train[w2v_train['label']==0]
df_minor=w2v_train[w2v_train['label']==1]
df_minor_upsampled = resample(df_minor, 
                                 replace=True,     # sample with replacement
                                 n_samples=major_class_0)
df_w2v_upsampled = pd.concat([df_major, df_minor_upsampled])
print('shape',df_w2v_upsampled.shape)
sns.countplot(df_w2v_upsampled.label)


# # Split Dataset

# In[ ]:


x=df_bow_upsampled.iloc[:,0:-1]
y=df_bow_upsampled['label']
x_train_bow,x_test_bow,y_train_bow,y_test_bow=train_test_split(x,y,test_size=0.2)


# In[ ]:


x=df_tfidf_upsampled.iloc[:,0:-1]
y=df_tfidf_upsampled['label']
x_train_tfidf,x_test_tfidf,y_train_tfidf,y_test_tfidf=train_test_split(x,y,test_size=0.2)


# In[ ]:


x=df_w2v_upsampled.iloc[:,0:-1]
y=df_w2v_upsampled['label']
x_train_w2v,x_test_w2v,y_train_w2v,y_test_w2v=train_test_split(x,y,test_size=0.2)


# # Model Selection

# In[ ]:


def f1_score_(y_proba,y_test):
  proba = y_proba[:,1] >= 0.3
  proba = proba.astype(np.int) 
  return f1_score( proba,y_test)   


# ## RandomForest

# In[ ]:


#use Bow
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100)
model.fit(x_train_bow,y_train_bow)
y_pred=model.predict(x_test_bow)
acc=accuracy_score(y_pred,y_test_bow)
print('Accuracy Score',acc)
accuracy.append(acc)
y_proba=model.predict_proba(x_test_bow)
f1_scor=f1_score_(y_proba,y_test_bow)
print('f1 score ',f1_scor)


# In[ ]:


#use tfidf
model=RandomForestClassifier(n_estimators=100)
model.fit(x_train_tfidf,y_train_tfidf)
y_pred=model.predict(x_test_tfidf)
acc=accuracy_score(y_pred,y_test_tfidf)
print('Accuracy Score',acc)
y_proba=model.predict_proba(x_test_tfidf)
f1_scor=f1_score_(y_proba,y_test_tfidf)
print('f1 score ',f1_scor)


# In[ ]:


#use word2vec
model=RandomForestClassifier(n_estimators=100)
model.fit(x_train_w2v,y_train_w2v)
y_pred=model.predict(x_test_w2v)
acc=accuracy_score(y_pred,y_test_w2v)
print('Accuracy Score',acc)
y_proba=model.predict_proba(x_test_w2v)
f1_scor=f1_score_(y_proba,y_test_w2v)
print('f1 score ',f1_scor)


# # Summary
# <table>
# <tr>
# <td colspan=2>BOW</td>
# <td colspan=2>TF-IDF</td>
# <td colspan=2>WORD2VEC</td>
# </tr>
#  <tr>
# <td>Accuray</td>
# <td>f1_score</td>
# <td>Accuray</td>
# <td>f1_score</td>
# <td>Accuray</td>
# <td>f1_score</td>
# </tr>
# <tr>
# <td>0.9733344549125168</td>
# <td>0.9462331732661575</td>
# <td>0.9733344549125168</td>
# <td>0.9494853523357087</td>
# <td>0.993606998654105</td>
# <td>0.9792162983652345</td>
# </tr>
# 
# </table>
# 
# 

# In[ ]:




