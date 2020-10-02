#!/usr/bin/env python
# coding: utf-8

# ![Imgur](https://i.imgur.com/iy82iZq.png)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import zipfile

files=['/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip',
       '/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip',
       '/kaggle/input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip']

for file in files :
    zip = zipfile.ZipFile(file,'r')
    zip.extractall()
    zip.close()


# In[ ]:


train=pd.read_csv('/kaggle/working/labeledTrainData.tsv', delimiter="\t")
test=pd.read_csv('/kaggle/working/testData.tsv', delimiter="\t")


# In[ ]:


sub=pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/sampleSubmission.csv')


# ## 1. EDA of review texts
# 1. `Character distriubtion` of each review
# 1. `Word distriubtion` of each review
# 1. `Word cloud` of each word
# 1. Distribution by `Sentiment class`
# 1. Ratio with `special characters`

# In[ ]:


train.head()


# In[ ]:


print('the train data is : {} line'.format(len(train)))
print('the test data is : {} line'.format(len(test)))


# In[ ]:


train_len=train['review'].apply(len)
test_len=test['review'].apply(len)

import matplotlib.pyplot as plt
import seaborn as sns
fig=plt.figure(figsize=(15,4))
fig.add_subplot(1,2,1)
sns.distplot((train_len),color='red')

fig.add_subplot(1,2,2)
sns.distplot((test_len),color='blue')


# In[ ]:


train['word_n'] = train['review'].apply(lambda x : len(x.split(' ')))
test['word_n'] = test['review'].apply(lambda x : len(x.split(' ')))

fig=plt.figure(figsize=(15,4))
fig.add_subplot(1,2,1)
sns.distplot(train['word_n'],color='red')

fig.add_subplot(1,2,2)
sns.distplot(test['word_n'],color='blue')


# In[ ]:


train['length']=train['review'].apply(len)
train['length'].describe()


# In[ ]:


train['word_n'].describe()


# - `Distribution of words in one review` is similar both in train and test set
# - The `mean words` count is 233 and `std` is 173 words
# - The character count seems to show similar distribution with word count

# In[ ]:


from wordcloud import WordCloud
cloud=WordCloud(width=800, height=600).generate(" ".join(train['review'])) # join function can help merge all words into one string. " " means space can be a sep between words.
plt.figure(figsize=(15,10))
plt.imshow(cloud)
plt.axis('off')


# - `br` is the most frequent one. But br is a sort of HTML tag, Thus it should be removed.
# - `movie` or `film` is the theme which all reviews share. Thus I suppose `idf(inverse document frequency)` shoul be close to zero

# In[ ]:


fig, axe = plt.subplots(1,3, figsize=(23,5))
sns.countplot(train['sentiment'], ax=axe[0])
sns.boxenplot(x=train['sentiment'], y=train['length'], data=train, ax=axe[1])
sns.boxenplot(x=train['sentiment'], y=train['word_n'], data=train, ax=axe[2])


# - The distribution of sentiment is `half and half` between zero and one
# - The review length distribution by sentiment is similar but if somebody feels harshly dissatisfied, the reveiw tends to be wordy (more outliers)

# In[ ]:


print('the review with question mark is {}'.format(np.mean(train['review'].apply(lambda x : '?' in x))))
print('the review with fullstop mark is {}'.format(np.mean(train['review'].apply(lambda x : '.' in x))))
print('the ratio of the first capital letter is {}'.format(np.mean(train['review'].apply(lambda x : x[0].isupper()))))
print('the ratio with the capital letter is {}'.format(np.mean(train['review'].apply(lambda x : max(y.isupper() for y in x)))))
print('the ratio with the number is {}'.format(np.mean(train['review'].apply(lambda x : max(y.isdigit() for y in x)))))


# ## 2. Preprocessing
# 1. Remove `HTML tags` such as `<br>` using BeautifulSoup
# 1. Only `english character` will remain using regular expression
# 1. By NLTK, `stopwords` will be eliminated

# In[ ]:


import re
import json
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


# In[ ]:


train['review']=train['review'].apply(lambda x: BeautifulSoup(x,"html5lib").get_text())
test['review']=test['review'].apply(lambda x: BeautifulSoup(x,"html5lib").get_text())


# In[ ]:


train['review']=train['review'].apply(lambda x: re.sub("[^a-zA-Z]"," ",x))
test['review']=test['review'].apply(lambda x: re.sub("[^a-zA-Z]"," ",x))


# In[ ]:


train.head(3)


# In[ ]:


stops = set(stopwords.words("english"))

for i in range(0,25000) : 
    review = train.iloc[i,2] # review column : 2 
    review = review.lower().split()
    words = [r for r in review if not r in stops]
    clean_review = ' '.join(words)
    train.iloc[i,2] = clean_review


# In[ ]:


for i in range(0,25000) : 
    review = test.iloc[i,1] # review column : 1
    review = review.lower().split()
    words = [r for r in review if not r in stops]
    clean_review = ' '.join(words)
    test.iloc[i,1] = clean_review


# In[ ]:


train['word_n_2'] = train['review'].apply(lambda x : len(x.split(' ')))
test['word_n_2'] = test['review'].apply(lambda x : len(x.split(' ')))

fig, axe = plt.subplots(1,1, figsize=(7,5))
sns.boxenplot(x=train['sentiment'], y=train['word_n_2'], data=train)


# - After preprocessing, the distribution by sentiment in train data is `not so different` from previous state `except total counts`

# In[ ]:


from keras.preprocessing.text import Tokenizer
tk = Tokenizer()
tk.fit_on_texts(list(train['review'])+list(test['review']))
text_seq_tr=tk.texts_to_sequences(train['review'])
text_seq_te=tk.texts_to_sequences(test['review'])
word_ind=tk.word_index


# - Usiung keras, tokenization and mapping to numbers are done
# - When fitting, `use all data from train and text data set`, which prevents model from errors

# In[ ]:


print('Total word count is :',len(word_ind))


# In[ ]:


data_info={}
data_info['word_ind']=word_ind
data_info['word_len']=len(word_ind)+1


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

fig=plt.figure(figsize=(15,4))
fig.add_subplot(1,2,1)
sns.distplot(pd.Series(text_seq_tr).apply(lambda x : len(x)))
fig.add_subplot(1,2,2)
sns.distplot(pd.Series(text_seq_te).apply(lambda x : len(x)))


# In[ ]:


from keras.preprocessing.sequence import pad_sequences
pad_train=pad_sequences(text_seq_tr, maxlen=400) 
pad_test=pad_sequences(text_seq_te, maxlen=400) 


# - `max length` is set, if length more than max length, `zero value` will replace that place

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(pad_train, train['sentiment'], random_state=77, test_size=0.07, stratify=train['sentiment'])


# - use validation set, when we make a model. test_size is set in between 5% to 10%, to use more data

# In[ ]:


len(tk.word_index)


# ## 3. Modeling
# 1. *`sequential model`* using adam optimizer
# 1. set `early stopping` and `model checkpoint` (patient option)

# In[ ]:


from keras import Sequential
from keras.layers import Dense, Embedding, Flatten

model=Sequential()
model.add(Embedding(101247,65, input_length=400))
model.add(Flatten())
model.add(Dense(2,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'] )


# In[ ]:


from keras.callbacks import EarlyStopping, ModelCheckpoint
es=EarlyStopping(patience=4) 
mc=ModelCheckpoint('best.h5',save_best_only=True)
model.fit(x_train,y_train, batch_size=128, epochs=10, validation_data=[x_valid,y_valid], callbacks=[es,mc]) 


# In[ ]:


model.load_weights('best.h5')


# In[ ]:


res=model.predict(pad_test, batch_size=128)


# In[ ]:


res


# In[ ]:


sub['sentiment_pro']=res[:,1]


# In[ ]:


sub.loc[sub['sentiment_pro']>=0.5,"sentiment"]=1
sub.loc[sub['sentiment_pro']<0.5,"sentiment"]=0


# - Use *`0.5 as thereshold`* to specify one or zero

# In[ ]:


sub=sub[['id','sentiment']]


# In[ ]:


sub.to_csv('result.csv',index=False)

