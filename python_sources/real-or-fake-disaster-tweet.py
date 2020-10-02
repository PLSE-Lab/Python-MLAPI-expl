#!/usr/bin/env python
# coding: utf-8

# # Import libs

# In[ ]:


get_ipython().system('pip install -q swifter # faster pandas apply')


# In[ ]:


import numpy as np 
import pandas as pd
import swifter 
import seaborn as sns
import re
from sklearn.model_selection import StratifiedKFold
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Input, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
tqdm.pandas()


# Let see the given data

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")


# # Data Exploration

# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# Get country

# In[ ]:


import geopy
import pycountry
import math
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="navneet")

def get_location(region):
    if pd.isnull(region):
        return None
    try:
        return geolocator.geocode(region)[0].split(",")[-1] 
    except:
        return "no_country"


# In[ ]:


train_df["country"] = train_df["location"].swifter.progress_bar(enable=True).apply(get_location)


# In[ ]:


test_df["country"] = test_df["location"].swifter.progress_bar(enable=True).apply(get_location)


# In[ ]:


nan_values = {"keyword": "no_keyword", "country": "no_location"}
train_df.fillna(value=nan_values, inplace=True)
test_df.fillna(value=nan_values, inplace=True)
train_df.head()


# Check class balance by our target

# In[ ]:


sns.countplot(train_df.target)


# Check on null columns

# In[ ]:


train_df.isnull().sum()


# There are no null in text and target. But we have 61 null in keywords and 2533 null in location

# ### 1. Explor keywords

# In[ ]:


sns.countplot(y=train_df.keyword,  order=train_df.keyword.value_counts()[:20].index, orient='v')


# In[ ]:


total_keywords = len(train_df.keyword.unique())
total_locations = len(train_df.country.unique())


# # One Hot Encoding

# In[ ]:


keyword_encoder = OneHotEncoder()
keyword_encoder.fit(train_df.keyword.values.reshape(-1,1))
keyword_encoder.fit(test_df.keyword.values.reshape(-1,1))
encoded_keywords = keyword_encoder.transform(train_df.keyword.values.reshape(-1,1)).toarray()


# In[ ]:


location_encoder = OneHotEncoder()
location_encoder.fit(train_df.country.values.reshape(-1,1))
location_encoder.fit(test_df.country.values.reshape(-1,1))
encoded_locations = location_encoder.fit_transform(train_df.country.values.reshape(-1,1)).toarray()


# 
# # Text Cleaning

# In[ ]:


# https://stackoverflow.com/a/47091490
def decontracted(sentence):
    """Convert contractions like "can't" into "can not"
    """
    # specific
    sentence = re.sub(r"won\'t", "will not", sentence)
    sentence = re.sub(r"can\'t", "can not", sentence)

    # general
    #phrase = re.sub(r"n't", " not", phrase) # resulted in "ca not" when sentence started with "can't"
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'s", " is", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'t", " not", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'m", " am", sentence)
    return sentence


# In[ ]:


# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# In[ ]:


tokenizer = RegexpTokenizer(r'\w+')
def remove_punctuation(text):
    text = tokenizer.tokenize(text)
    return text


# In[ ]:


def remove_stopwords(text):
    filtered_words = [w for w in text if not w in stopwords.words('english')]
    return " ".join(filtered_words)


# In[ ]:


def clean_urls(text):
    return re.sub('https?://\S+|www\.\S+', '', text)


# In[ ]:


def clean_tweet(text):
    text = text.lower()
    text = clean_urls(text)
    text = decontracted(text)
    text = remove_emoji(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    return text


# In[ ]:


train_df['text'] = train_df['text'].progress_apply(clean_tweet)


# In[ ]:


test_df['text'] = test_df['text'].progress_apply(clean_tweet)


# # Text Tokenization

# In[ ]:


MAX_NB_WORDS=225
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(train_df['text'])
tokenizer.fit_on_texts(test_df['text'])
sequences_data = tokenizer.texts_to_sequences(train_df['text'])


# Total words

# In[ ]:


word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# pad sequences to 225

# In[ ]:


MAX_SEQUENCE_LENGTH=225
sequences_data = pad_sequences(sequences_data, maxlen=MAX_SEQUENCE_LENGTH)


# # Preparing the Embedding layer

# In[ ]:


embeddings_index = {}
f = open("/kaggle/input/embeddings/glove-840B-300d.txt", encoding='latin')
for line in f:
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float16')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))


# Create embedding Matrix

# In[ ]:


EMBEDDING_DIM = 300 # glove-840B-300d
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# # Embedding Layer

# In[ ]:


embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


# # Our Model

# In[ ]:


def build_model():
    keyword_input = Input(shape=(total_keywords,), dtype='int32')
    location_input = Input(shape=(total_locations,), dtype='int32')
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    
    embedded = embedding_layer(sequence_input)
    embedded = Conv1D(64, 5, activation='relu')(embedded)
    embedded = MaxPooling1D(5)(embedded)
    embedded = Conv1D(32, 5, activation='relu')(embedded)
    embedded = MaxPooling1D(5)(embedded)
    embedded = Flatten()(embedded)
    
    x = concatenate([keyword_input, location_input, sequence_input])
    x = Dense(512, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)

    model = Model([keyword_input, location_input, sequence_input], preds)

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model


# # Cross Validation

# In[ ]:


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train, test in kfold.split(sequences_data, train_df.target):
    model = build_model()

    train_sequences = sequences_data[train]
    train_keywords = encoded_keywords[train]
    train_locations = encoded_locations[train]
    train_target = train_df.target[train]
    
    test_sequences = sequences_data[test]
    test_keywords = encoded_keywords[test]
    test_locations = encoded_locations[test]
    test_target = train_df.target[test]
    
    model.fit([train_keywords, train_locations, train_sequences], train_target)
    loss, acc = model.evaluate([test_keywords, test_locations, test_sequences], test_target)
    print("Loss: ", loss)
    print("Accuracy: ", acc)


# # Train Model

# In[ ]:


model = build_model()
model.fit([encoded_keywords, encoded_locations, sequences_data], train_df.target)


# # Prediction

# In[ ]:


sequences_data = tokenizer.texts_to_sequences(test_df['text'])
sequences_data = pad_sequences(sequences_data, maxlen=MAX_SEQUENCE_LENGTH)
encoded_keywords = keyword_encoder.transform(test_df.keyword.values.reshape(-1, 1)).toarray()
encoded_locations = location_encoder.transform(test_df.country.values.reshape(-1, 1)).toarray()
predictions = model.predict([encoded_keywords, encoded_locations, sequences_data])


# # Submission

# In[ ]:


submission = pd.DataFrame(
    {
        'id': test_df.id,
        'target': predictions.flatten()
    }, 
    columns = ['id', 'target']
)


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv("submission.csv")

