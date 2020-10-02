#!/usr/bin/env python
# coding: utf-8

# I tried to learn Deep learning sentiment analysis using the notbook of https://www.kaggle.com/paoloripamonti/twitter-sentiment-analysis

# In[ ]:


# DataFrame
import pandas as pd

# Matplot
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns
sns.set(style='darkgrid')

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# nltk
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

# Word2vec
import gensim

# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import time
import pickle
import itertools

# Set log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


nltk.download('stopwords')


# In[ ]:


DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
df = pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv', encoding =DATASET_ENCODING , names=DATASET_COLUMNS)
df.head()


# In[ ]:


df.shape


# In[ ]:


mapping = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
def decode_sentiment(label):
    return mapping[int(label)]
df.target = df.target.apply(lambda x: decode_sentiment(x))


# In[ ]:


df.head()


# In[ ]:


## Plotting the bar char to identify the frequnecy of values
sns.countplot(df.target)
##prinitng number of values for each type
print(df.target.value_counts())


# In[ ]:


stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)


# In[ ]:


df.text = df.text.apply(lambda x: preprocess(x))
df=df.filter(['target','text'], axis=1)


# In[ ]:


df_train, df_test = train_test_split(df, test_size=1-0.8, random_state=42)
print("TRAIN size:", len(df_train))
print("TEST size:", len(df_test))


# In[ ]:


documents = [_text.split() for _text in df_train.text] 


# <h5>Gensim Word2VEc paramters </h5>
# size: The number of dimensions of the embeddings and the default is 100. <br/>
# window: The maximum distance between a target word and words around the target word. The default window is 5.<br/>
# min_count: The minimum count of words to consider when training the model; words with occurrence less than this count will be ignored. The default for min_count is 5.<br/>
# workers: The number of partitions during training and the default workers is 3.<br/>

# In[ ]:


w2v_model = gensim.models.word2vec.Word2Vec(size=300, 
                                            window=5, 
                                            min_count=5, 
                                            workers=3)


# In[ ]:


w2v_model.build_vocab(documents)


# In[ ]:


words = w2v_model.wv.vocab.keys()
vocab_size = len(words)
print("Vocab size", vocab_size)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'w2v_model.train(documents, total_examples=len(documents), epochs=20)')


# In[ ]:


w2v_model.most_similar("love")


# In[ ]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train.text)

vocab_size = len(tokenizer.word_index) + 1
print("Total words", vocab_size)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'SEQUENCE_LENGTH=300\nx_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=SEQUENCE_LENGTH)\nx_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=SEQUENCE_LENGTH)')


# In[ ]:


POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
labels = df_train.target.unique().tolist()
labels.append(NEUTRAL)
encoder = LabelEncoder()
encoder.fit(df_train.target.tolist())

y_train = encoder.transform(df_train.target.tolist())
y_test = encoder.transform(df_test.target.tolist())

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print("y_train",y_train.shape)
print("y_test",y_test.shape)


# In[ ]:




