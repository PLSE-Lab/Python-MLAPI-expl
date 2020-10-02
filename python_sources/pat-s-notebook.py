#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#LOAD DATA
print("Loading data...")
train = pd.read_table("../input/train.tsv")
test = pd.read_table("../input/test.tsv")
print(train.shape)
print(test.shape)


# In[ ]:


from matplotlib import pyplot as plt


# In[ ]:


plt.hist(train['item_condition_id'])
plt.show()


# In[ ]:


plt.hist(train['price'],bins=range(0,200,5)),plt.show()


# In[ ]:


def handle_missing(dataset):
    dataset.category_name.fillna(value="empty", inplace=True)
    dataset.brand_name.fillna(value="empty", inplace=True)
    dataset.item_description.fillna(value="empty", inplace=True)
    return (dataset)

train = handle_missing(train)
test = handle_missing(test)
print(train.shape)
print(test.shape)
y_train=train['price']


# In[ ]:



import pandas as pd
from sklearn.manifold import TSNE
from sklearn import preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


# ## text cleaning

# In[ ]:


noise_list = ["is", "a", "this", "...","/n","the","I","you","at","in","an","of","to"]
stopwords.words('english').append(noise_list)
noise=stopwords.words('english').append(noise_list)
for i in stopwords.words('english'):
    noise_list.append(i)
noise_list=set(noise_list)

lem = WordNetLemmatizer()

def cleanShitUp(x):
    def dropper(text):
        words=text.split()
        new_words=[word.replace('?','').replace('!','').replace(',','').replace(';','').replace('\"',"")                   .replace('\'',"") for word in words]
        newer_words=[word for word in new_words if len(word)>1]
        new_text=" ".join(newer_words)
        return new_text

    def _remove_noise(input_text):
        words = input_text.split()
        noise_free_words = [word for word in words if word not in noise_list]
        noise_free_text = " ".join(noise_free_words)
        return noise_free_text

    def lemmer(text):
        words=text.split()
        new_words=[lem.lemmatize(word,"v") for word in words]
        new_text=" ".join(new_words)
        return new_text
    
    cleaned=dropper(x)
    cleaned=lemmer(cleaned)
    cleaned= _remove_noise(cleaned)
    return cleaned


# In[ ]:


train['cat_clean']=train['category_name'].apply(lambda x: cleanShitUp(x.lower()))
train['brand_clean']=train['brand_name'].apply(lambda x: cleanShitUp(x.lower()))
train['desc_clean']=train['item_description'].apply(lambda x: cleanShitUp(x.lower()))
test['cat_clean']=test['category_name'].apply(lambda x: cleanShitUp(x.lower()))
test['brand_clean']=test['brand_name'].apply(lambda x: cleanShitUp(x.lower()))
test['desc_clean']=test['item_description'].apply(lambda x: cleanShitUp(x.lower()))


# In[ ]:


y_train=train['price']


# In[ ]:


cat_avgs=train[['cat_clean','price']].groupby(['cat_clean'], as_index=False).mean()
cat_avgs.columns = ['cat_clean', 'cat_price']


# In[ ]:


empty_avg=float(cat_avgs[cat_avgs['cat_clean']=='empty']['cat_price'])
empty_avg


# In[ ]:


brand_avgs=train[['brand_clean','price']].groupby(['brand_clean'], as_index=False).mean()
brand_avgs.columns = ['brand_clean', 'brand_price']


# In[ ]:


empty_avg_brand=float(brand_avgs[brand_avgs['brand_clean']=='empty']['brand_price'])
empty_avg_brand


# In[ ]:


submit2.to_csv('sample_submission2.csv',index=False)


# In[ ]:


new_train=train.merge(brand_avgs,on='brand_clean',how='left').merge(cat_avgs,on='cat_clean',how='left')

#.drop(train.columns[['brand_name_brand', 'category_name_cat']], axis=1)


# In[ ]:


new_test=test.merge(brand_avgs,on='brand_clean',how='left').merge(cat_avgs,on='cat_clean',how='left')


# In[ ]:


new_train['cat_price'].fillna(value=round(empty_avg,2), inplace=True)
new_train['brand_price'].fillna(value=empty_avg_brand, inplace=True)
new_test['cat_price'].fillna(value=round(empty_avg,2), inplace=True)
new_test['brand_price'].fillna(value=empty_avg_brand, inplace=True)


# In[ ]:


pd.set_option('display.max_columns', 500)


# In[ ]:


new_train.columns,new_test.columns


# In[ ]:


from keras.preprocessing.text import Tokenizer
raw_text = np.hstack([new_train.desc_clean.str.lower(), new_train.name.str.lower()])

print("   Fitting tokenizer...")
tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)
print("   Transforming text to seq...")
train_array_desc= tok_raw.texts_to_sequences(new_train.desc_clean.str.lower())
test_array_desc = tok_raw.texts_to_sequences(new_test.desc_clean.str.lower())
train_array_name= tok_raw.texts_to_sequences(new_train.name.str.lower())
test_array_name= tok_raw.texts_to_sequences(new_test.name.str.lower())


# In[ ]:


new_train_array_desc=[]
for i,j in enumerate(train_array_desc):
    new_train_array_desc.append([k for k in train_array_desc[i] if k < 5000])


# In[ ]:


new_test_array_desc=[]
for i,j in enumerate(new_test_array_desc):
    new_test_array_desc.append([k for k in new_test_array_desc[i] if k < 5000])


# In[ ]:


new_train.desc_clean.str.lower()


# In[ ]:


import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('price',axis=1),                                                    y_train, test_size=0.33, random_state=0)


# In[ ]:


X_train_keras,X_test_keras, y_train_keras,y_test_keras=train_test_split(new_train_array_desc,                                                    y_train, test_size=0.33, random_state=0)


# In[ ]:


X_train_keras = sequence.pad_sequences(X_train_keras, maxlen=100)
X_test_keras= sequence.pad_sequences(X_test_keras, maxlen=100)


# In[ ]:


def rmsle_cust(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))


# In[ ]:


# create the model
tot_words= 5000 
embedding_vector_length = 25
model = Sequential()
model.add(Embedding(tot_words, embedding_vector_length, input_length=100))
model.add(LSTM(50))
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
print(model.summary())
model.fit(X_train_keras, y_train_keras,          validation_data=(X_test_keras, y_test_keras), epochs=3, batch_size=128)


# In[ ]:


import pickle
pickle_out = open("model_desc.pickle","wb")
pickle.dump(model, pickle_out)
pickle_out.close()


# In[ ]:


scores = model.evaluate(X_test_keras, y_test_keras, verbose=0)


# # Next steps
# ## Train lstm on name and categories (split on "/")
# ## Combine 3 lstms prediction with other features, including cat avg, brand avg, len(item_desc)...
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




