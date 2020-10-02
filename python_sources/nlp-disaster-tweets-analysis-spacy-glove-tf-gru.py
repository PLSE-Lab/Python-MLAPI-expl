#!/usr/bin/env python
# coding: utf-8

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


import seaborn as sb
import matplotlib.pyplot as plt
import re
import nltk
import spacy
import string
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# models
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, GRU
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf


# In[ ]:


pd.options.display.max_colwidth = 200  # set a value as required for better visualization


# In[ ]:


get_ipython().system('ls -lh /kaggle/input/glovetwitter27b200dtxt')


# # Introduction
# 
# * Here I will analyze training and test data, will understand its features. 
# * As normally twitter texts are unprocessed, so I will clean the data using various tools like regex, spaCy etc.
# * Will use GloVe for pretrained vectors.
# * Finally, I will train a model using deep learning (Tensorflow).
# * Will do the prediction using the trained mode.
# 
# I will try to keep the implementation simple with explanations in each step.
# 
# I have learned from various sources including https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub, please upvote it also.
# For anything unclear or any suggestion for improvement of the kernel please do let me know.
# 
# Thanks
# 
# Arijit

# # Check the training and test data source

# In[ ]:


# tweet train data
twdata=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
# test data
testdata=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')


# Check the training and test data features 

# In[ ]:


twdata.head()


# In[ ]:


testdata.head()


# In[ ]:


# check some random sample to get some more understanding on the keyword & location. 
# I am just showing one random sample, but to understand the data better I have checked many samples.
twdata.sample(n=5)


# In[ ]:


# check the NaN count in various features.
twdata.isnull().sum()


# In[ ]:


testdata.isnull().sum()


# We need to remove the NaN from the datsets, this will be done in the next section.

# In[ ]:


print('training data shape {}, test data shape {}'.format(twdata.shape,testdata.shape))


# # Data Analysis

# **location" does not contain valid data and in many places data is missing. So will drop this feature.**

# *Analyze the **"keyword"** feature

# In[ ]:


def plot_feature_col(data, col, comment):
    keyword_dist=data[col].value_counts()

    fig = plt.figure(figsize=(16, 3), dpi=100) # figsize-width,height

    keyword_dist[:80].plot.bar() # take the top 80

    plt.title(comment)
    plt.legend([col+' count'])
    plt.show()
    
    return keyword_dist


# In[ ]:


# check the keyword distribution
keyword_dist_training = plot_feature_col(twdata, 'keyword', 'keyword Distribution in training data')


# **Check the test data keyword distribution**

# In[ ]:


# check the keyword distribution using plot
keyword_dist_test = plot_feature_col(testdata, 'keyword', 'keyword Distribution in test data')


# **Analysis:**
# 
# From the above its observed that many places keyword's are allready present in the text and some time not.
# Also many keyword's are common in training & test data.
# **Will combine the keyword and text features for further analysis**

# In[ ]:


# for keyword fill the NaN values as NAN, will remove the same later
twdata['keyword'].fillna('NAN', inplace=True)
testdata['keyword'].fillna('NAN', inplace=True)


# In[ ]:


# combine keyword & text
twdata['text']=twdata['text'] + " " + twdata['keyword']
testdata['text']=testdata['text'] + " " + testdata['keyword']


# In[ ]:


# Now I will drop location & keyword.
twdata.drop(columns=['location','keyword'],axis=1, inplace=True)
testdata.drop(columns=['location','keyword'],axis=1, inplace=True)

Check the training & test data
# In[ ]:


twdata.head(2)


# In[ ]:


testdata.head(2)


# In[ ]:


twdata.isnull().sum()


# In[ ]:


testdata.isnull().sum()


# **lets check the 'target' to understand its distribution.**

# In[ ]:


target_dist_training = plot_feature_col(twdata, 'target', 'target distribution in training data')


# **Decision:**
# 
# there is no big gap between two different kind of taget types(i.e. 1 & 0), will go with this as it is

# # Remove noise from data
# In the train & test data both consists of lot of noise (stop words, delimeters, special symbols, lower case etc.), that need to be removed before we process it further.

# In[ ]:


# load spacy for data cleaning
nlp = spacy.load('en_core_web_sm')


# In[ ]:


# tokenize 
twdata['clean_text']=twdata['text'].apply(lambda x: list(nlp(x)))


# In[ ]:


# function to clean the text data using spaCy
def spacy_clean_text(text):
    # remove punctuations 
    text = [t for t in text if (t.is_punct == False)]

    # remove stopwords
    text = [t for t in text if (t.is_stop == False)]
    
    # remove digits
    text = [t for t in text if (t.is_digit == False)]
    
    # lemmatize (should be done at the end)
    text = [t.lemma_ for t in text]
    
    # join it to get back the original sentence
    text = " ".join(text)

    # convert to lower case
    text = text.lower()

    return text


# In[ ]:


twdata['clean_text'] = twdata['clean_text'].apply(lambda x: spacy_clean_text(x))


# In[ ]:


# check the cleaned data
twdata.sample(5)


# In[ ]:


# clean text data still has http://, @..., so need to clean further
def clean_regex(text):
    # remove http://
    text = re.sub(r'http\S+', '', text)
    # remove '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    text = ''.join([x for x in text if x not in string.punctuation])
    # remove non ascii
    text = ''.join([x for x in text if ord(x) < 128])

    return text


# In[ ]:


twdata['clean_text'] = twdata['clean_text'].apply(lambda x: clean_regex(x))


# In[ ]:


twdata.sample(5)


# In[ ]:


# check the clean data length distribution
clean_text_length=[len(x) for x in twdata['clean_text']] 
sb.distplot(clean_text_length, axlabel='clean_text length', color="g")


# In[ ]:


twdata.shape


# In[ ]:


# rt & nan is still there
def clean_regex_next(text):
    # remove rt
    text = re.sub(r'rt', '', text)
    # remove nan
    text = re.sub(r'nan', '', text)
    # remove digits
    text = "".join(t for t in text if not t.isdigit())
   
    return text


# In[ ]:


twdata['clean_text'] = twdata['clean_text'].apply(lambda x: clean_regex_next(x))


# In[ ]:


twdata.head()


# In[ ]:


# lets plot to understand the most common word's
all_data= " ".join(twdata['clean_text'])
words=all_data.split()


sb.set(rc={'figure.figsize':(20,5)})

nltk_plot=nltk.FreqDist(words)
nltk_plot.plot(100)


# **In the above its visible that there are few single charecters, need to drop them.**

# In[ ]:


def drop_single_char(text):
    text =  text.split()
    text = " ".join(t for t in text if len(t)>1)

    return text


# In[ ]:


twdata['clean_text'] = twdata['clean_text'].apply(lambda x: drop_single_char(x))


# In[ ]:


# there should be no single charecters now
all_data=" ".join(twdata['clean_text'])
words=all_data.split()

sb.set(rc={'figure.figsize':(20,5)})

nltk_plot=nltk.FreqDist(words)
nltk_plot.plot(100)


# Now training data is ready for creating a model.

# ** similar way clean the test data also **

# In[ ]:


testdata['clean_text']=testdata['text'].apply(lambda x: list(nlp(x)))
testdata['clean_text'] = testdata['clean_text'].apply(lambda x: spacy_clean_text(x))


# In[ ]:


testdata['clean_text'] = testdata['clean_text'].apply(lambda x: clean_regex(x))
testdata['clean_text'] = testdata['clean_text'].apply(lambda x: clean_regex_next(x))
testdata['clean_text'] = testdata['clean_text'].apply(lambda x: drop_single_char(x))


# In[ ]:


# verify the test data
testdata.head()


# In[ ]:


# check the clean text length distribution for test data
clean_text_length=[len(x) for x in testdata['clean_text']] 
sb.distplot(clean_text_length, axlabel='test clean_text length', color="g")


# In[ ]:


testdata.shape


# In[ ]:


# plot the word frequency to understand the test data
all_data_test=" ".join(testdata['clean_text'])
words_test=all_data_test.split()

sb.set(rc={'figure.figsize':(20,5)})

nltk_plot=nltk.FreqDist(words_test)
nltk_plot.plot(100)


# In[ ]:


testdata.sample(5)


# Some sentence in training and test data consists of only one word. If they are not useful it has to be dropped.

# In[ ]:


# find single word in training data sentences
single_word_train=[]
for sent in twdata.clean_text:
  if len(sent.split())==1:
    single_word_train.append(sent)

print(single_word_train)


# All the above records need to be dropped as they are not adding any value in the training data.

# In[ ]:


for idx in twdata.index:
    row_series=twdata.loc[idx]
    if len(row_series['clean_text'].split())==1:
        twdata.drop(idx, inplace=True) # drop th erecord


# In[ ]:


twdata.shape


# In[ ]:


# find single word in test data sentences
single_word_test=[]
for sent in testdata.clean_text:
  if len(sent.split())==1:
    single_word_test.append(sent)

print(single_word_test)


# this are also not usefull for test data, but record cant be dropped so the submission length will go wrong.
# Insted we can add this data as record in training data for creating model, then model will take care of the same while doing prediction.

# In[ ]:


df2 = pd.DataFrame(columns=['id', 'text', 'target', 'clean_text'],
                  data=[[7607, 'hey',0, 'hey'],
                        [7608, 'fuck',0, 'fuck'],
                        [7609, 'nooooooooo',0, 'nooooooooo'],
                        [7610, 'tell',0, 'tell'],
                        [7611, 'awesome',0, 'awesome']])
df2.head()


# In[ ]:


twdata = twdata.append(df2)
twdata.shape


# In[ ]:


twdata.tail()


# In[ ]:


# save the cleaned data for future processing.
twdata.to_pickle('train_clean.pickle')
testdata.to_pickle('test_clean.pickle')


# # Create Model with Tensorflow 

# **Lets use the GloVe vectors and prepare the word embedding**
# 
# will use the http://nlp.stanford.edu/data/glove.twitter.27B.zip data for the pretrained vectors.

# In[ ]:


# create the word embedding index.
embeddings_index = dict()
f = open('../input/glovetwitter27b200dtxt/glove.twitter.27B.200d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


# In[ ]:


# join the train and test data to tokenize.
train_rows=twdata['clean_text'].shape[0]
test_rows=testdata['clean_text'].shape[0]

total_data=pd.concat([twdata, testdata], sort=False)
print('total date shape {}'.format(total_data.shape))


# In[ ]:


# tokenize the total data
vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(total_data['clean_text'])

sequences = tokenizer.texts_to_sequences(total_data['clean_text'])
data = pad_sequences(sequences, maxlen=20)


# In[ ]:


# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocabulary_size, 200))
for word, index in tokenizer.word_index.items():
    if index > vocabulary_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector


# In[ ]:


model_glove = Sequential()
# word embedding, trainable should be False.
model_glove.add(Embedding(vocabulary_size, 200, input_length=20, weights=[embedding_matrix], trainable=False))

model_glove.add(Dropout(0.2)) # to avoid overfitting.
model_glove.add(Conv1D(128, 5, activation='relu'))
model_glove.add(MaxPooling1D(pool_size=4))

# using GRU we can do training faster than LSTM.
model_glove.add(GRU(128,dropout=0.1, recurrent_dropout=0.2))

# final dense layers
model_glove.add(Dense(32, activation='relu'))
model_glove.add(Dense(16, activation='relu'))
model_glove.add(Dense(8, activation='relu'))

#outout layer.
model_glove.add(Dense(1, activation='sigmoid'))


# In[ ]:


model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model_glove.summary()


# In[ ]:


# create early stop criteria and save the best model.
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=2)  
mc = ModelCheckpoint('best_model.h5', monitor='loss', mode='min', save_best_only=True,verbose=1)


# In[ ]:


# lets train the model.
batch_size = 128
num_epochs = 20

history = model_glove.fit(data[:train_rows, :], twdata['target'], batch_size = batch_size, epochs = num_epochs, callbacks=[es,mc])


# In[ ]:


# plot the training loss and accuracy to understand how the model has performed.
history_dict = history.history

acc = history_dict['accuracy']
loss=history_dict['loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(8,6))
plt.plot(epochs, loss, 'b', label='loss')
plt.plot(epochs, acc, 'r', label='accuracy')

plt.title('Training loss & accuracy')
plt.xlabel('loss')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# **Use the model for doing the prediction**

# In[ ]:


from tensorflow.keras.models import load_model

# load best model
model = load_model('best_model.h5')
y_pred = model.predict(data[train_rows:, :])


# In[ ]:


sample = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv", nrows=test_rows)
sample['target'] = np.round(y_pred).astype('int')
sample.to_csv('model_submission.csv', index=False)


# In[ ]:


# lets manually check the prediction value and compare with test data.
sample.head(20)


# In[ ]:


testdata.head(20)


# In[ ]:


get_ipython().system('kaggle competitions submit -c nlp-getting-started -f model_submission.csv -m "Message"')


# # create model with Google BERT

# In[ ]:


# We will use the official tokenization script created by the Google team
get_ipython().system('wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py')


# In[ ]:


import tensorflow_hub as hub

import tokenization
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# In[ ]:


def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


# In[ ]:


def build_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# In[ ]:


get_ipython().run_cell_magic('time', '', 'module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"\nbert_layer = hub.KerasLayer(module_url, trainable=True)')


# In[ ]:


vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)


# In[ ]:


train_input = bert_encode(twdata.text.values, tokenizer, max_len=160)
test_input = bert_encode(testdata.text.values, tokenizer, max_len=160)
train_labels = twdata.target.values


# In[ ]:


model = build_model(bert_layer, max_len=160)
model.summary()


# In[ ]:


checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint],
    batch_size=16
)


# In[ ]:


from tensorflow.keras.models import load_model

# load best model
model.load_weights('model.h5')
test_pred = model.predict(test_input)


# In[ ]:


test_pred[:10]


# In[ ]:


sample = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample['target'] = test_pred.round().astype(int)
sample[:20]


# In[ ]:


testdata[:20]


# In[ ]:


sample.to_csv('model_submission.csv', index=False)


# In[ ]:




