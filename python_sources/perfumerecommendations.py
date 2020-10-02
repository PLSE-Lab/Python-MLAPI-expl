#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
from keras.models import Model
from keras.layers import Dense, Flatten, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM
import gc
import keras
from keras.layers import Dense, Concatenate
from keras.layers import *
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras import losses
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras import optimizers
from itertools import product
from keras.models import load_model


# In[ ]:


cust_df = pd.read_csv("/kaggle/input/parisdata/profilesniche_prepared.csv")
perf_df = pd.read_csv("/kaggle/input/parisprodaccords/products_finals_with_accords.csv")
ratings_df = pd.read_csv("/kaggle/input/kernel70b3382677/ratings_copy.csv")


# In[ ]:


cust_df.shape


# In[ ]:


cust_df.head()


# In[ ]:


cust_df = cust_df[['IDcustomer', 'text']]


# In[ ]:


cust_df.isna().sum()


# In[ ]:


cust_df['text'].fillna("unknown", inplace = True)


# In[ ]:


perf_df.shape


# In[ ]:


perf_df.head()


# In[ ]:


perf_df[['0', '1', '2', '3']]


# In[ ]:


#removeing names infromt of nodes

vals = ['Top0', 'Top1', 'Top2', 'Top3', 'Middle0', 'Middle1', 'Middle2']
for i in ['0', '1', '2', '3']:
    print(i)
    for val in vals:
        perf_df[i] = perf_df[i].apply(lambda x: x.replace(val,''))


# In[ ]:


perf_df['0'].nunique(), perf_df['1'].nunique(), perf_df['2'].nunique(), perf_df['3'].nunique(), 


# In[ ]:


perf_df[['0', '1', '2', '3']].tail()


# In[ ]:


perf_df[['url', '0', '1', '2', '3']].to_csv("perfume_nodes.csv", index=None)


# In[ ]:


dummy = list(perf_df['0']) + list(perf_df['1']) +  list(perf_df['2']) +  list(perf_df['3']) 


# In[ ]:


dummies = set(dummy)


# In[ ]:


len(dummies)


# In[ ]:


nodes_encoding = {i:j for j, i in enumerate(dummies)}


# In[ ]:


nodes_encoding['Cetalox']


# In[ ]:


perf_df['Top0'] =  perf_df['0'].apply(lambda x:nodes_encoding[x])
perf_df['Top1'] =  perf_df['1'].apply(lambda x:nodes_encoding[x])
perf_df['Top2'] =  perf_df['2'].apply(lambda x:nodes_encoding[x])
perf_df['Top3'] =  perf_df['3'].apply(lambda x:nodes_encoding[x])

del perf_df['0']
del perf_df['1']
del perf_df['2']
del perf_df['3']


# In[ ]:


perf_df.head()


# In[ ]:


del perf_df['title']


# In[ ]:


perf_df.head()


# In[ ]:


cust_df.head()


# In[ ]:


ratings_df.head()


# In[ ]:


for col in ['avatar_img', 'date', 'karma', 'name_perfume', 'rew', 'username']:
    del ratings_df[col]


# In[ ]:


ratings_df.head()


# In[ ]:


def get_user_id(x):
    vals = re.findall(r'\d+', x)
    return vals[0]

ratings_df['IDcustomer'] = ratings_df['userlink'].apply(lambda x: get_user_id(x))


# In[ ]:


del ratings_df['userlink']


# In[ ]:


ratings_df.head()


# In[ ]:


cust_df.head()


# In[ ]:


perf_df.head()


# In[ ]:


url_encoding = {}
for i, j in enumerate(perf_df['url']):
    url_encoding[j] = i


# In[ ]:


perf_df['ID_perfume'] = perf_df['url'].apply(lambda x:url_encoding[x])


# In[ ]:


perf_df.head()


# In[ ]:


ratings_df.head()


# In[ ]:


gc.collect()


# In[ ]:


def get_url_encoding(x):
    try:
        vals = url_encoding[x]        
    except:
        vals = None
    return vals


# In[ ]:


ratings_df['ID_perfume'] = ratings_df['url'].apply(lambda x: get_url_encoding(x))


# In[ ]:


ratings_df.tail()


# In[ ]:


ratings_df.isna().sum()


# In[ ]:


ratings_df = ratings_df[ratings_df['ID_perfume'].isna()== False]


# In[ ]:


ratings_df.shape


# In[ ]:


del ratings_df['url']
del perf_df['url']

del ratings_df['text']
# In[ ]:


cust_df.head()


# In[ ]:


perf_df.head()


# In[ ]:


ratings_df.head()


# In[ ]:


ratings_df['ID_perfume'] = ratings_df['ID_perfume'].astype('int')


# In[ ]:


cust_df.shape, perf_df.shape, ratings_df.shape


# In[ ]:


ratings_df['IDcustomer'].dtypes, cust_df['IDcustomer'].dtypes 


# In[ ]:


ratings_df['IDcustomer'] = ratings_df['IDcustomer'].astype('int')


# In[ ]:


ratings_df.head()


# In[ ]:


del ratings_df['text']


# In[ ]:


existing_df = pd.merge(ratings_df, cust_df, on='IDcustomer', how='left')


# In[ ]:


existing_df = pd.merge(existing_df, perf_df, on='ID_perfume', how='left')


# In[ ]:


existing_df.shape


# In[ ]:


existing_df.head()


# In[ ]:


existing_df.isna().sum()


# In[ ]:


existing_df['text'].fillna('unknown', inplace = True)


# In[ ]:


existing_df.head()


# In[ ]:


combination_existed = existing_df[['IDcustomer', 'ID_perfume']]


# In[ ]:


dummy = existing_df['text'].apply(lambda x :len(x.split()))


# In[ ]:


dummy.describe()


# In[ ]:


def clean_text(x):
    x = x.lower()
    x = re.sub('[^A-Za-z0-9]+', ' ', x)
    return x


# In[ ]:


existing_df['text'] = existing_df['text'].apply(lambda x:clean_text(x))


# In[ ]:


gc.collect()


# In[ ]:


## some config values 
embed_size = 300 # how big is each word vector
max_features = 100000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 150 # max number of words in a question to use


# In[ ]:


my_list = [i for i in existing_df['text']]


# In[ ]:


my_words = " ".join(i for i in my_list)


# In[ ]:


words_length = len(set(my_words.split()))
words_length


# In[ ]:


max_features = words_length


# In[ ]:





# In[ ]:


## fill up the missing values
train_X = existing_df['text'].fillna("##").values

print("before tokenization")
print(train_X.shape)


## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))

train_X = tokenizer.texts_to_sequences(train_X)

print("after tokenization")
print(len(train_X))


# In[ ]:


train_X = pad_sequences(train_X, maxlen=maxlen)


# In[ ]:


gc.collect()


# In[ ]:


existing_df.head()


# In[ ]:


x_train = existing_df.drop(['sentiment', 'IDcustomer',  'ID_perfume', 'text'], axis=1)


# In[ ]:


x_train.shape


# In[ ]:


train_X.shape


# In[ ]:


import numpy as geek 
#geek.save('tokenisedfile', train_X) 


# In[ ]:


#np.savetxt('textembeddings.txt', train_X)


# In[ ]:


gc.collect()


# In[ ]:


x_train.shape, train_X.shape, existing_df['sentiment'].shape


# In[ ]:


np.random.seed(0)

indices = np.random.permutation(x_train.shape[0])
training_idx, test_idx = indices[:700000], indices[700000:]


# In[ ]:





# In[ ]:


x_train, x_test = x_train.iloc[training_idx,:], x_train.iloc[test_idx,:]

x_train_embed, x_test_embed = train_X[training_idx,:], train_X[test_idx,:]


# In[ ]:


x_train.shape, x_test.shape


# In[ ]:


target = existing_df['sentiment'].values


# In[ ]:


target


# In[ ]:


target_encoded = np.where(target>0.6, 1, 0)


# In[ ]:


target_encoded


# In[ ]:


y_train, y_test = target_encoded[training_idx], target_encoded[test_idx]


# In[ ]:


y_train.shape, y_test.shape


# In[ ]:


gc.collect()


# In[ ]:


x_train_embed.shape, x_train.shape


# In[ ]:


inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)
#lstm = Bidirectional(LSTM(200, dropout=0.2, recurrent_dropout=0.2))(x)
#x = Embedding(max_features, embed_size)(inp)

x = LSTM(256, return_sequences=True)(x)
#x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
x = LSTM(64, return_sequences=True)(x)
x = Flatten()(x)

agei = Input(shape=(153,))
#agei = Dense(100)(agei)

conc = concatenate([x, agei])

drop = Dropout(0.2)(conc)
dens = Dense(100)(drop)
dens = Dense(1)(dens)
acti = Activation('sigmoid')(dens)

model = Model(inputs=[inp, agei], outputs=acti)
#optimizer = optimizers.Adam(lr=1e-4)
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics = ['acc'])


# In[ ]:


#SVG(model_to_dot(model).create(prog='dot', format='svg'))


# In[ ]:


model.fit([x_train_embed, x_train], y_train, validation_data=([x_test_embed, x_test], y_test),epochs=1,
          batch_size=1024, shuffle=True, verbose=1)


# In[ ]:


model.save('my_model.h5') 


# In[ ]:


model.save('my_model_new.hdf5') 


# In[ ]:


#predicted = model.predict([x_train_embed, x_train])


# In[ ]:


#predicted[:100]


# In[ ]:


#target[training_idx][:100]


# In[ ]:


#existing_df.head()


# In[ ]:


gc.collect()


# In[ ]:


#existing_df.to_csv("existing_df.csv", index = None)


# In[ ]:


gc.collect()


# In[ ]:


my_list = list(product(existing_df['IDcustomer'][:10], existing_df['ID_perfume'].unique()))
newdf = pd.DataFrame(data=my_list, columns=['IDcustomer','ID_perfume'])


# In[ ]:


newdf['IDcustomer'].nunique()


# In[ ]:


newdf['IDcustomer']


# In[ ]:


newdf['combo'] = newdf['IDcustomer'].apply(str) + " " + newdf['ID_perfume'].apply(str)
combination_existed['combo'] = combination_existed['IDcustomer'].apply(str) + " " + combination_existed['ID_perfume'].apply(str)


# In[ ]:


combination_existed['combo'].values


# In[ ]:


test_df = newdf.loc[~newdf['combo'].isin(combination_existed['combo'].values)]


# In[ ]:


del test_df['combo']


# In[ ]:


gc.collect()


# In[ ]:


del existing_df
del newdf
del combination_existed


# In[ ]:


gc.collect()


# In[ ]:


test_df = pd.merge(test_df, cust_df, on='IDcustomer', how='left')
test_df = pd.merge(test_df, perf_df, on='ID_perfume', how='left')


# In[ ]:


test_df.shape


# In[ ]:


test_X = test_df['text'].fillna("##").values

test_X = tokenizer.texts_to_sequences(test_X)

test_X = pad_sequences(test_X, maxlen=maxlen)


# In[ ]:


test_df.head()


# In[ ]:


x_test = test_df.drop(['IDcustomer',  'ID_perfume', 'text'], axis=1)


# In[ ]:


x_test.shape


# In[ ]:


test_X.shape


# In[ ]:


y_pred = model.predict([test_X, x_test])


# In[ ]:


y_pred[:5]


# In[ ]:





# In[ ]:


test_df['prediction'] = y_pred


# In[ ]:


test_df.head()


# In[ ]:


def get_recommendations(cust_id):
    results = test_df[test_df['IDcustomer']==cust_id][['ID_perfume', 'prediction']]
    return results.sort_values(by ='prediction' , ascending=False)[:10]


# In[ ]:


get_recommendations(1141379)


# In[ ]:





# In[ ]:




