#!/usr/bin/env python
# coding: utf-8

# This probably is the dataset that is closest to my heart. Because its about food :-), more precisely about the cuisines of the world and ingredients.
# 
# 
# ![](https://media.giphy.com/media/MXWQKVEi2OM5W/giphy.gif)

# In[ ]:


import os
import numpy as np 
import pandas as pd 
from pandas import read_csv

print(os.listdir("../input"))
pd.set_option('display.max_colwidth', -1)
np.random.seed(7)


# <h3> Data level Challenges </h3>

# 
# The datasets are JSON files. The raw json object is nested and it looks like below.
# So data level we have some challenge.  We need to flatten the nested json style ingredients and see this huge blob as a NLP feature to fit a model to the data. Let's see how do we do that.
# ```json
# [  
#    {  
#       "id":10259,
#       "cuisine":"greek",
#       "ingredients":[  
#          "romaine lettuce",
#          "black olives",
#          "grape tomatoes",
#          "garlic",
#          "pepper",
#          "purple onion",
#          "seasoning",
#          "garbanzo beans",
#          "feta cheese crumbles"
#       ]
#    },
#    {  
#       "id":25693,
#       "cuisine":"southern_us",
#       "ingredients":[  
#          "plain flour",
#          "ground pepper",
#          "salt",
#          "tomatoes",
#          "ground black pepper",
#          "thyme",
#          "eggs",
#          "green tomatoes",
#          "yellow corn meal",
#          "milk",
#          "vegetable oil"
#       ]
#    }
# ]
# ```
# 

# <h3> Problem domain Challenges </h3>

# 
# 
# The challenge is while we know different international cuisines share ingredients, given a set of ingredients how accuractely can you predict the cuisine ? The challenge has two parts, 
# <ul>
# <li>There are lot of frequently occuring ingredients which may **not** be predictive like **salt, garlic, pepper, milk or tomatoes**. </li>
# <li>But there can be **pretty unique and predictive** ingredients like **black olives and feta cheese**. </li>
#  </ul>
#  
#  In a NLP sense, salt and garlic are like the "stop words" of this corpus. Should we construct a special stop word list to remove the less predictive ingredients ?

# Ok, Let's start with flattening and exploring the data.  Also, we need to attend to a tiny little detail here, if we flatted all the ingredients to form blob say, like this "romaine lettuce black olives grape tomatoes" we lose the affinity between the word n-grams. What do I mean by that black olives is actually a single ingredient and our model should have some means to look at it as single word "black_olives"  (i.e as a bigram) and not as black olives. So can we as a part of flattening all the data make sure all the word n-grams are intact and not losing their  affinity ? Thats where 1D-CNN comes into play. They are a natural  **dynamic**  n-gram detector.  
# 
# 
# Next, do you think word count can be predictive of the cuisine ? Say Indian cuisine would use more ingredients than Italian or Greek hence bigger word count  ? I don't know, sounds like a decent numerical feature to test with :-)

# In[ ]:


train_filename =  '../input/train.json'
test_filename =  '../input/test.json'

def flatten_json(input_file):
    import json
    import re
    from pandas.io.json import json_normalize
    corpus_file = open(input_file,"r")
    corpus = corpus_file.read()
    entries =  json.loads(corpus)
    df =  json_normalize(entries)
    df['flat_ingredients'] = df.apply(lambda row: ' '.join(ingredient for ingredient in row['ingredients']), axis=1)
    df['word_count'] = df.apply(lambda row: len(row['flat_ingredients'].split(' ')), axis=1)
    df.drop('ingredients', axis=1, inplace=True)   
    df.sort_values(['word_count'], ascending=False, inplace=True)
    return df                          
        
train_data_raw = flatten_json(train_filename)
test_data_raw = flatten_json(test_filename)


# In[ ]:


train_data_raw.head()


# Word count could be predictive, But I was dead wrong about *Italian cuisine , the cuisine that topped with most ingredients is Italian with 65 ingredients and Brazilian seconds it ??. I wanna know what these dishes are !* . Here I continue shamelessly. 

# <h3> TF-IDF + Logistic Regression </h3>

# In[ ]:


# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer 
# from sklearn.model_selection import cross_val_score 
# from scipy.sparse import hstack

# ingredient_vectorizer = TfidfVectorizer(
#                         sublinear_tf=True,
#                         strip_accents='unicode',
#                         analyzer='word',
#                         #token_pattern=r'\w{1,}',
#                         ngram_range=(1,4),
#                         stop_words='english',
#                         max_features=200000)

# ingredient_char_vectorizer = TfidfVectorizer(
#                     sublinear_tf=True,
#                     strip_accents='unicode',
#                     analyzer='char',
#                     stop_words='english',
#                     ngram_range=(5, 10),
#                     max_features=400000)

# all_ingredients = pd.concat([
#      train_data_raw['flat_ingredients'],
#      test_data_raw['flat_ingredients']]
# )

# ingredient_vectorizer.fit(all_ingredients)
# ingredient_char_vectorizer.fit(all_ingredients)

# # Create TF-IDF vectors for training features.
# x_word_train = ingredient_vectorizer.transform(train_data_raw['flat_ingredients'])
# x_char_train = ingredient_char_vectorizer.transform(train_data_raw['flat_ingredients'])
# x_train = hstack([x_char_train, x_word_train])
# # Create TF-IDF vectors for test features.
# x_word_test = ingredient_vectorizer.transform(test_data_raw['flat_ingredients'])
# x_char_test = ingredient_char_vectorizer.transform(test_data_raw['flat_ingredients'])
# x_test = hstack([x_char_test, x_word_test])

# y_train = pd.get_dummies(train_data_raw['cuisine'])
# columns  = y_train.columns.tolist()


# In[ ]:


# output = pd.DataFrame()
# scores = []
# for column in columns:
#     classifier = LogisticRegression(C=0.1,  solver='sag')
#     classifier.fit(x_train, y_train[column])
#     output[column] = classifier.predict_proba(x_test)[:, 1]
#     cv_score = np.mean(cross_val_score(classifier, x_train, y_train[column], cv=3, scoring='accuracy'))
#     scores.append(cv_score)
#     print('CV score for class {} is {}'.format(column, cv_score))

# print('Total CV score is {}'.format(np.mean(scores)))


# In[ ]:


# cuisine = output.idxmax(axis=1)
# cuisine.shape
# submission = pd.DataFrame.from_dict({'id': test_data_raw['id'],'cuisine': cuisine})
# submission.to_csv('submission.csv',index=False)
# submission.head()


# <h3> biLSTM + CNN </h3>

# In[ ]:


from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

vocab_size =  50000            # based on words in the entire corpus
max_len = 64                   # based on word count in phrases

all_corpus     = list(train_data_raw['flat_ingredients'].values) + list(test_data_raw['flat_ingredients'].values)
train_phrases  = list(train_data_raw['flat_ingredients'].values) 
test_phrases   = list(test_data_raw['flat_ingredients'].values)
X_train_target_binary = pd.get_dummies(train_data_raw['cuisine'])
columns = X_train_target_binary.columns.tolist()

#Vocabulary-Indexing of the train and test flat_ingredients, make sure "filters" parm doesn't clean out punctuations
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(all_corpus)
word_index = tokenizer.word_index
print("word_index", len(word_index))

encoded_train_phrases = tokenizer.texts_to_sequences(train_phrases)
encoded_test_phrases = tokenizer.texts_to_sequences(test_phrases)

#Watch for a POST padding, as opposed to the default PRE padding
X_train_words = sequence.pad_sequences(encoded_train_phrases, maxlen=max_len,  padding='post')
X_test_words = sequence.pad_sequences(encoded_test_phrases, maxlen=max_len,  padding='post')
print (X_train_words.shape)
print (X_test_words.shape)
print (X_train_target_binary.shape)

X_train_num = train_data_raw['word_count']
X_test_num = test_data_raw['word_count']

print ('Done Tokenizing and indexing phrases based on the vocabulary learned from the entire Train and Test corpus')


# In[ ]:


from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import  GlobalMaxPool1D, SpatialDropout1D
from keras.layers import Bidirectional
from keras.models import Model
from keras.layers.merge import concatenate

early_stop = EarlyStopping(monitor = "val_loss", mode="min", patience = 3, verbose=1)

print("Building layers")        
nb_epoch = 10
print('starting to stitch and compile  model')
from keras.initializers import he_normal
initializer = he_normal(seed=None)

input_numeric = Input((1, ))
dense_num = Dense(32, activation='relu')(input_numeric)

# Embedding layer for text inputs
input_words = Input((max_len, ))
x_words = Embedding(vocab_size, 300, input_length=max_len)(input_words)
x_words = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x_words)
x_words = Dropout(0.5)(x_words)
x_words = Conv1D(64, 3,   activation='relu')(x_words)
x_words = Conv1D(64, 1,   activation='relu')(x_words)
x_words = GlobalMaxPool1D()(x_words)
x_words = Dropout(0.5)(x_words)

# merge
merged = concatenate([x_words, dense_num])
x = Dense(64, activation="relu")(merged)
predictions = Dense(20, activation="softmax")(x)

model = Model(inputs=[input_words, input_numeric], outputs=predictions)
model.compile(optimizer='nadam',loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())


# In[ ]:


#fit the model
history = model.fit([X_train_words, X_train_num], X_train_target_binary, epochs=nb_epoch, verbose=1, batch_size = 128, callbacks=[early_stop], validation_split = 0.2, shuffle=True)
train_loss = np.mean(history.history['loss'])
val_loss = np.mean(history.history['val_loss'])
print('Train loss: %f' % (train_loss))
print('Validation loss: %f' % (val_loss))


# In[ ]:


pred = model.predict([X_test_words,X_test_num], batch_size=128, verbose = 1)
print (pred.shape) 
max_pred = np.round(np.argmax(pred, axis=1)).astype(int)
cuisines = [columns[m] for m in max_pred]
df =pd.DataFrame({'cuisines': cuisines}).reset_index()
df.groupby('cuisines').agg('count')
submission = pd.DataFrame({'id':test_data_raw['id'],'cuisine': cuisines})
submission.to_csv('submission.csv',index=False)

    

