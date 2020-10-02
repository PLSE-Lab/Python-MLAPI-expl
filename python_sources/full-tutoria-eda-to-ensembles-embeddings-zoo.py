#!/usr/bin/env python
# coding: utf-8

# ![](https://breakingtech.it/wp-content/uploads/2018/04/twitter-moments-1.jpg)
# 
# ## What's in the notebook?
# - Full Exploratory Data Analysis (EDA)
# - Data Cleaning
# - Evaluation
#     - BL Models (majority model + tfidf & logreg)
#     - Gradient Boosting
#     - Simple RNN
#     - Glove Bi-LSTM
#     - BERT + sigmoid
#     - Ensemble (BERT + 10 shallow classifiers)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import  Counter
plt.style.use('ggplot')
stop=set(stopwords.words('english'))
import re
from nltk.tokenize import word_tokenize
import gensim
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from sklearn.preprocessing import LabelEncoder
## ADD STOPWORDS
stop = set(list(stop) + ["http","https", "s", "nt", "m"])


# In[ ]:


def load_training(training_path="/kaggle/input/nlp-getting-started/train.csv"):
    df = pd.read_csv(training_path)
    
    print(df.head(10))
    return df

df = load_training()


# # Data Analisys
# In the following we're gonna see some data analysis on the corpus. 
# 
# Specifically:
# - General dataset infos
#     - Number of samples
#     - Data Columns 
#     - Class Label Distributiom
# - Text analysis
#     - Number of characters in tweets
#     - Number of words in a tweet
#     - Average word lenght in a tweet
#     - Word distribution
#     - Hashtag Analysis
#     - KW and Location Analysis

# ## General dataset information
# Here we show the number of samples, the input data columns and the class label distributiom

# In[ ]:


print("-Number of samples: {}".format(len(df)))
print("-Input data columns: {}".format(df.columns))
print("-Class label distribution")
print("--Number of positive samples: {}".format(len(df.loc[df['target'] == 1])))
print("--Number of negative samples: {}".format(len(df.loc[df['target'] == 0])))
print("--Plot of Y distributions")
x=df.target.value_counts()
sns.barplot(x.index,x)
plt.gca().set_ylabel('samples')

def plot_hist_classes(to_plot, _header):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
    df_len = to_plot(1)
    ax1.hist(df_len,color='red')
    ax1.set_title('Negative Tweets [disasters]')
    df_len = to_plot(0)
    ax2.hist(df_len,color='green')
    ax2.set_title('Positive Tweets [good posts]')
    fig.suptitle(_header)
    plt.show()
    plt.close()
    


# ## Text analysis
# Insights on number of character and words in tweets, word lenght distribution, and word distribution.

# In[ ]:


def show_word_distrib(target=1, field="text"):
    txt = df[df['target']==target][field].str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
    words = nltk.tokenize.word_tokenize(txt)
    words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stop) 
    
    rslt = pd.DataFrame(words_except_stop_dist.most_common(top_N),
                        columns=['Word', 'Frequency']).set_index('Word')
    print(rslt)
    matplotlib.style.use('ggplot')

    rslt.plot.bar(rot=0)
    
print("-Number of characters in tweets")
def to_plot(_target):
    return df[df['target']==_target]['text'].str.len()

    
plot_hist_classes(to_plot, _header='Characters Distribution in Tweets')


print("-Number of words in a tweet")
def to_plot(_target):
    return df[df['target']==_target]['text'].str.split().map(lambda x: len(x))
def how_to_plot(**kwargs):
    ax1.hist(df_len,**kwargs)
plot_hist_classes(to_plot, _header='Word Distribution in Tweet')


print("-Average word lenght in a tweet")
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
word=df[df['target']==1]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='red')
ax1.set_title('Negative Tweets [disasters]')
word=df[df['target']==0]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='green')
ax2.set_title('Positive Tweets [good posts]')
fig.suptitle('Average word length in each tweet')
plt.show()

print("-Word distribution")

top_N = 10

print("-- Positive Class")

show_word_distrib(target=1, field="text")

print("-- Negative Class")
show_word_distrib(target=0, field="text")


# ## Hashtag analysis
# Small analysis done on the hashtags, to check it's possible discriminator capability for this task.

# In[ ]:


print("-Hashtag Analysis ")
def find_hashtags(tweet):
    return ", ".join([match.group(0)[1:] for match in re.finditer(r"#\w+", tweet)]) or None

def add_hashtags(df):
    from sklearn.feature_extraction.text import CountVectorizer
    
    df['hashtag'] = df["text"].apply(lambda x: find_hashtags(x))
    df['hashtag'].fillna(value="no", inplace=True)
    
    return df
    
top_N = 20

df = add_hashtags(df)
_l = len([v for v in df.hashtag.values if isinstance(v, str)])
print("-Number of tweets with hashtags: {}".format(_l))
print("-- Hashtag distribution in positive samples ")
show_word_distrib(target=1, field="hashtag")

print("-- Hashtag distribution in negative samples ")
show_word_distrib(target=0, field="hashtag")


#There is too much intersection between hashtag in positive and negative samples, meaning that an
#hashtag approach will not work that well.


# # KW and Location analysis
# 

# In[ ]:


#Remove the encoded space character for keywords, since appears a lot of times and is junk
df['keyword'] = df['keyword'].map(lambda s: s.replace('%20', ' ') if isinstance(s, str) else s)

un_KW  = {kw for kw in df['keyword' ].values if isinstance(kw, str)}
tot_KW = len(df) - len(df[df["keyword" ].isna()])

un_LOC = {lc for lc in df['location'].values if isinstance(lc, str)}
tot_LOC =  len(df) - len(df[df["location"].isna()])

print("Unique KW: {}".format(len(un_KW)))
print("Out of: {}".format(tot_KW))
print("Samples with no KW: {}".format(len(df[df['keyword'].isna()])))


print("Unique LOC: {}".format(len(un_LOC)))
print("Out of: {}".format(tot_LOC))
print("Samples with no Loc: {}".format(len(df[df['location'].isna()])))


#LOCATION IS TOO SPARSE TO BE USED.


# In[ ]:


disaster_keywords = [kw for kw in df.loc[df.target == 1].keyword]
regular_keywords = [kw for kw in df.loc[df.target == 0].keyword]

disaster_keywords_counts = dict(pd.DataFrame(data={'x': disaster_keywords}).x.value_counts())
regular_keywords_counts = dict(pd.DataFrame(data={'x': regular_keywords}).x.value_counts())

all_keywords_counts =  dict(pd.DataFrame(data={'x': df.keyword.values}).x.value_counts())

# we sort the keywords so the most frequents are on top and we print them with relative
# occurrences in both classes of tweets:

for keyword, _ in sorted(all_keywords_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print("> KW: {}".format(keyword))
    print("-- # in negative tweets: {}".format(disaster_keywords_counts.get(keyword, 0)))
    print("-- # in positive tweets: {}".format(regular_keywords_counts.get(keyword, 0)))
    print('--------')


#Many KWs in negative tweets are also present in positive ones, meaning that a KW approach is most likely to not work
#The same result was given by the hashtag analysis.


# # Data Cleaning
# Here we are gonna clean the DF.
# Specifically, we clean:
# - stopwords (Kept cause removing them cause drop of performances)
# - URL 
# - HTML 
# - emoji 
# - punctuation

# In[ ]:



def clean_df(df):
    def remove_stopwords(text):
        if text is not None:
            tokens = [x for x in word_tokenize(text) if x not in stop]
            return " ".join(tokens)
        else:
            return None
    
    #TMP: TRY TO USE DEFAULT STRING FOR NONE. TODO: USE ROW["KEYWORDS"]
    #df['hashtag'] =df['hashtag'].apply(lambda x : "NO" if x is None else x)
    
    df["text"] = df['text'].apply(lambda x : x.lower())
    #df["hashtag"] = df['hashtag'].apply(lambda x : x.lower())
    
    #df['text'] =df['text'].apply(lambda x : remove_stopwords(x))
    #df['hashtag'] =df['hashtag'].apply(lambda x : remove_stopwords(x))
    
    
    


    def remove_URL(text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    df['text']=df['text'].apply(lambda x : remove_URL(x))
    def remove_html(text):
        html=re.compile(r'<.*?>')
        return html.sub(r'',text)

    df['text']=df['text'].apply(lambda x : remove_html(x))
    # Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
    def remove_emoji(text):
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    df['text']=df['text'].apply(lambda x: remove_emoji(x))
    def remove_punct(text):
        table=str.maketrans('','',string.punctuation)
        return text.translate(table)

    df['text']=df['text'].apply(lambda x : remove_punct(x))
    
    df.text = df.text.replace('\s+', ' ', regex=True)
    return df
df = clean_df(df)
print("-- Word distrig Positive Class")

show_word_distrib(target=1, field="text")

print("-- Word distrib Negative Class")
show_word_distrib(target=0, field="text")


# # Utils for models

# In[ ]:


def read_test(test_path="/kaggle/input/nlp-getting-started/test.csv"):
    
    my_df = pd.read_csv(test_path)
    
    res_df = my_df[['id']]
    my_df = my_df[['text']]
    
    add_hashtags(my_df)
    my_df = clean_df(my_df)
    print("Test DF: {}".format(my_df.head(10)))
    
    return my_df, res_df

def dump_preds(res_df, preds, out="default"):
    res_df['target'] = None
    
    for i, p in  enumerate(preds):
        res_df.ix[i, 'target'] = p
    
    res_df.to_csv(out, index = False)
    

def split_data(df, _t=True):
    X = df.text
    if _t:
        Y = df.target
        le = LabelEncoder()
        Y = le.fit_transform(Y)
        Y = Y.reshape(-1,1)
        return X, Y
    else:
        return X

    


# # Baseline Models
# We build a simple majority model and TFIDF + LogReg to check the problem hardness.

# In[ ]:


from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

"""
Build a baseline TFIDF + LOGREG based just on text
"""
def build_tfidf_logreg(df):
    my_df = df[['text','target']]
    x_features = my_df.columns[0]
    x_data = my_df[x_features]
    Y = my_df["target"]

    x_train, x_validation, y_train, y_validation = model_selection.train_test_split(
        x_data.values, Y.values, test_size=0.2, random_state=7)
    
    # configure TfidfVectorizer to accept tokenized data
    # reference http://www.davidsbatista.net/blog/2018/02/28/TfidfVectorizer/
    tfidf_vectorizer = TfidfVectorizer(
        analyzer='word',
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        token_pattern=None)

    lr = LogisticRegression()
    tfidf_lr_pipe = Pipeline([('tfidf', tfidf_vectorizer), ('lr', lr)])
    tfidf_lr_pipe.fit(x_train, y_train)
    
    return tfidf_lr_pipe

def test_tfidf_logreg(model, test_path="/kaggle/input/nlp-getting-started/test.csv"):
    
    my_df, res_df = read_test(test_path="/kaggle/input/nlp-getting-started/test.csv")
    
    #x_features = my_df.columns[0]
    x_data = my_df["text"].values

    preds = model.predict(x_data)
    
    #dump_preds(res_df, preds, out="res_tfidf_logreg4_0.csv")
    
    return res_df


"""
Build a majority model
"""
def test_majority_model(test_path="/kaggle/input/nlp-getting-started/test.csv"):
    
    my_df = pd.read_csv(test_path)
    
    res = my_df[['id']]
    res['target'] = 1
    
    res.to_csv("res_majority.csv", index = False)
    return res
    

#test_majority_model(test_path="/kaggle/input/nlp-getting-started/test.csv")
#0.42944

#tfidf_log_reg = build_tfidf_logreg(df)
#test_tfidf_logreg(tfidf_log_reg, test_path="/kaggle/input/nlp-getting-started/test.csv")
#0.63164





# # Test Gradient Boosting
# Here we check a gradient boosting classifier, which is a bit less shallow model w.r.t logistic regression. In fact we gain a 3% w.r.t LogReg

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

X_train, y_train = split_data(df)

test_df, res_df = read_test(test_path="/kaggle/input/nlp-getting-started/test.csv")
X_test = split_data(test_df, _t=False)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', GradientBoostingClassifier(n_estimators=100)),
                     ])
#text_clf.fit(X_train, y_train)
#predicted = text_clf.predict(X_test)
#dump_preds(res_df, predicted, out="submission.csv")
#0.66462


# In[ ]:





# # Test RNN Model
# Here we test a simple LSTM model with Dropout. The experiment does not give better performances w.r.t gradient boosting.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
get_ipython().run_line_magic('matplotlib', 'inline')

#Value tuned based on data analysis
max_words = 750
max_len = 160

def process_data(X, tok=None):
    if tok is None:
        tok = Tokenizer(num_words=max_words)
        tok.fit_on_texts(X)
    sequences = tok.texts_to_sequences(X)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
    return sequences_matrix, tok

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

def test_model(model, tok=None, test_path="/kaggle/input/nlp-getting-started/test.csv", cut=0.5):
    
    my_df, res = read_test(test_path="/kaggle/input/nlp-getting-started/test.csv")
    
    X = split_data(my_df, _t=False)
    sequences_matrix, tok = process_data(X, tok=tok)
    preds = model.predict(sequences_matrix)

    normalized_preds = []
    for p in preds:
        if p >= cut:
            normalized_preds.append(1)
            
        else:
            normalized_preds.append(0)
    #dump_preds(res, normalized_preds, out="res_rnn.csv")
    return res

def tune_cutoff(model, tok):
    
    x_tune = df['text'].values
    x_target = df[['target']]
    
    
    X, Y = split_data(df)
    sequences_matrix, tok = process_data(X, tok=tok)
    preds = model.predict(sequences_matrix)
    
    x_target["preds"] = preds
    
    accumulator = 0
    _0 = []
    max_0 = 0
    _1 = []
    min_1 = 0
    
    for i, row in x_target.iterrows():
        if row['target'] == 0:
            _0.append(row['preds'])
            if row['preds'] > max_0:
                max_0 = row['preds']
        else:
            _1.append(row['preds'])
            if row['preds'] < min_1:
                min_1 = row['preds']

    mean_0 = sum(_0)/len(_0)
    mean_1 = sum(_1)/len(_1)
    return max(mean_0, mean_1)/min(mean_0, mean_1)
    

#model = RNN()
#model.summary()
#model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
#X, Y = split_data(df)
#sequences_matrix, tok = process_data(X, tok=None)

#model.fit(sequences_matrix,Y,batch_size=128,epochs=10,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.001)])

    
#cut = tune_cutoff(model, tok)

#test_model(model, tok=tok, cut=cut)
#0.57259
#autocut: 0.57055


# # Test Glove LSTM
# Here we check the usage of Glove embeddings. We first encode the sentence with average word Glove embedding and then we use a Bi-LSTM to classify the sample representation. 
# We reach a new best of 79.45, meaning that the embedding direction seem to be valuable. We are gonna test some other newer embeddings.

# In[ ]:


import numpy as np 
import re  #regular expression
from bs4 import BeautifulSoup
import pandas as pd 
from sklearn import model_selection, preprocessing
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Embedding, Dropout, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

    
vocab_size = 10000
embedding_dim = 100
max_length = 50
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
validation_split = 0.10

def load_glove_vectors():
    print('Indexing word vectors.')
    #Many thanks to rtatman for hosting the GloVe word embeddings dataset on Kaggle
    #https://www.kaggle.com/rtatman/glove-global-vectors-for-word-representation
    GLOVE_DIR = '/kaggle/input/glove6b100dtxt/'
    embeddings_index = {}
    print(list(os.walk("/kaggle/input")))
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def tokenize(df):
    tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
    tokenizer.fit_on_texts(df.text)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    training_sequences = tokenizer.texts_to_sequences(df.text)
    training_padded = pad_sequences(training_sequences, maxlen = max_length, 
                                    padding = padding_type, truncating = trunc_type)
    print('Shape of the data vector is', training_padded.shape, df.target.shape)
    
    return training_sequences, training_padded, word_index, tokenizer

def prepare_embedding_layer(word_index, embeddings_index):
    print('Preparing the embedding matrix')
    num_words = min(vocab_size, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, index in word_index.items():
        if index >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word, np.zeros(embedding_dim, dtype='float32'))
        
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    embedding_layer = Embedding(num_words, embedding_dim, 
                           embeddings_initializer = Constant(embedding_matrix), 
                           input_length = max_length, 
                           trainable = False)
    
    
    return embedding_layer

def model(embedding_layer):
    
    METRICS = [
          metrics.BinaryAccuracy(name='accuracy'),
          metrics.Precision(name='precision'),
          metrics.Recall(name='recall'),
          metrics.AUC(name='auc')]

    sequence_input = Input(shape = (max_length, ))
    embedded_sequences = embedding_layer(sequence_input)
    x = Bidirectional(tf.keras.layers.LSTM(64))(embedded_sequences)
    x = Dropout(0.5)(x)
    x = Dense(32, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation = 'sigmoid')(x)
    model =  Model(sequence_input, output)
    model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(lr = .0002) ,metrics = METRICS)
    return model

def test_model(model, test_df, tokenizer):
    test_sequences = tokenizer.texts_to_sequences(test_df.text)
    test_padded = pad_sequences(test_sequences, maxlen = max_length, 
                                    padding = padding_type, truncating = trunc_type)
    predictions = model.predict(test_padded)
    predictions = np.round(predictions).astype(int).flatten()
    dump_preds(res, predictions, out="submission.csv")

"""
embeddings_index = load_glove_vectors()
test_df, res = read_test(test_path="/kaggle/input/nlp-getting-started/test.csv")
training_sequences, training_padded, word_index,tokenizer = tokenize(df)
embedding_layer =  prepare_embedding_layer(word_index, embeddings_index)

X_train, X_valid, y_train, y_valid = model_selection.train_test_split(training_padded, 
                                                                          df.target, 
                                                                          test_size = validation_split, 
                                                                          random_state=1)

model = model(embedding_layer)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)
history = model.fit(X_train, y_train, batch_size = 64, epochs = 30, 
                    callbacks = [early_stopping],
                    validation_data = (X_valid, y_valid))

test_model(model, test_df, tokenizer)
#0.7945
"""


# # BERT TfHub
# Using the BERT TfHub module, we build a DNN using BERT Embeddings and a simple Dense layer with sigmoid on the top. 
# This is the best scoring model so fa, reaching a best of 82.45%

# In[ ]:


get_ipython().system('wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py')


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Bidirectional,  Conv1D, MaxPooling1D
from tensorflow.keras.models import Model
import tensorflow_hub as hub

import tokenization

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

def build_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    
    """
    x = Bidirectional(tf.keras.layers.LSTM(64))(sequence_output)
    x = Dropout(0.1)(x)
    x = Dense(32, activation = 'relu')(x)
    x = Dropout(0.1)(x)
    out = Dense(1, activation = 'sigmoid')(x)
    #out = Dense(1, activation='sigmoid')(clf_output)
    """
    kernel_size = 5
    filters = 64
    pool_size = 4
    
    x = Dropout(0.1)(sequence_output)
    x = Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1)(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Bidirectional(tf.keras.layers.LSTM(64))(x)
    out = Dense(1, activation="sigmoid")(x)

    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

test_df, res = read_test(test_path="/kaggle/input/nlp-getting-started/test.csv")

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

train_input = bert_encode(df.text.values, tokenizer, max_len=160)
test_input = bert_encode(test_df.text.values, tokenizer, max_len=160)
train_labels = df.target.values

model = build_model(bert_layer, max_len=160)
model.summary()


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=4,
    mode='max',
    restore_best_weights=True)

train_history = model.fit(
    train_input, train_labels,
    validation_split=0.15,
    epochs=20,
    batch_size=16,
    callbacks=[early_stopping]
)

test_pred = model.predict(test_input)
_t = []
for t in test_pred:
    if t < 0.5:
        _t.append(0)
    else:
        _t.append(1)
dump_preds(res, _t, out="submission.csv")
#82.45


# # XLNet / RoBERTa in 4 lines!
# Here the XLNet/ GPT-2 / RoBERTa evaluation using the huggingface simpletransformers.

# In[ ]:


get_ipython().system('pip install simpletransformers')
get_ipython().system('git clone --recursive https://github.com/NVIDIA/apex.git')
get_ipython().system('pip install --upgrade --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" apex/. ')
import os
import shutil

shutil.rmtree('apex')
if os.path.exists("cache_dir"):
    shutil.rmtree('cache_dir')

if os.path.exists("outputs"):
    shutil.rmtree('outputs')

if os.path.exists("runs"):
    shutil.rmtree('runs')

import torch
import random 
import numpy as np

from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
seed = 98
"""
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

train_data = df[['text', 'target']]

print("Building Model")
train_args = {
    "train_batch_size": 16,
    'eval_batch_size': 4,
    "num_train_epochs": 10,
    "use_early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_delta": 0.005,
    'max_seq_length': 200,  
    'save_model_every_epoch': False,
    'overwrite_output_dir': True,
    'save_eval_checkpoints': False,
    "gradient_accumulation_steps": 1,
    "overwrite_output_dir": True
}
model = ClassificationModel("xlnet", 'xlnet-base-cased', num_labels=2, args=train_args)
#model = ClassificationModel("xlmroberta", 'xlm-roberta-base', num_labels=2, args=train_args)


print("Training the model")
model.train_model(train_data)


test_df, res = read_test(test_path="/kaggle/input/nlp-getting-started/test.csv")

print("Model evaluation")
predictions, raw_outputs = model.predict(test_df['text'])

dump_preds(res, predictions, out="submission_xlnet.csv")
"""


# # Test SuperLearner ensemble
# Finally we try a super-learner ensemble. Meaning that we use a set of models (also the previous BERT used) to classify the samples and then we use a meta-model to ingest these classification results (from all models) and classify the sample. 
# It is ensured that this ensemble cannot perform worse than the best in the ensemble, so we could enhance the BERT model capabilities.
# 

# In[ ]:


# example of a super learner model for binary classification

get_ipython().system('pip install tamnun')
from tamnun.bert import BertClassifier, BertVectorizer
from numpy import hstack
from numpy import vstack
from numpy import asarray
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.base import TransformerMixin 
from tamnun.bert import BertClassifier, BertVectorizer

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
import copy

class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()
    
# create a list of base-models
def get_models():
    models = []
    tfidf_vectorizer = TfidfVectorizer(
        analyzer='word',
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        token_pattern=None
    )
    lr = LogisticRegression()
    tfidf_lr_pipe = Pipeline([('tfidf', tfidf_vectorizer), ('lr', lr)])
    dec_pipe = Pipeline([('tfidf', tfidf_vectorizer), ('dt', DecisionTreeClassifier())])
    svc = Pipeline([('tfidf', tfidf_vectorizer), ('svc', SVC(gamma='scale', probability=True))])
    gaus = Pipeline([('tfidf', tfidf_vectorizer),('to_dense', DenseTransformer()),  ('gaus', GaussianNB())])
    kn = Pipeline([('tfidf', tfidf_vectorizer), ('kn', KNeighborsClassifier())])
    ada = Pipeline([('tfidf', tfidf_vectorizer), ('ada',AdaBoostClassifier())])
    bagging =  Pipeline([('tfidf', tfidf_vectorizer), ('bag',BaggingClassifier(n_estimators=10))])
    ran_forest = Pipeline([('tfidf', tfidf_vectorizer), ('ran',RandomForestClassifier(n_estimators=10))])
    extra_tree = Pipeline([('tfidf', tfidf_vectorizer), ('extr',ExtraTreesClassifier(n_estimators=10))])
    gradient_boosting = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', GradientBoostingClassifier(n_estimators=100)),
                     ])
    #bert =  Pipeline([('extr',BertVectorizer()), ("extr2",BertClassifier(num_of_classes=2))])
    
    module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
    bert_layer = hub.KerasLayer(module_url, trainable=True)

    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    model = build_model(bert_layer, max_len=160)
    model.summary()
    
   # models.append(bert)
    models.append(model)
    models.append(dec_pipe)
    models.append(tfidf_lr_pipe)
    models.append(svc)
    models.append(gaus)
    models.append(kn)
    models.append(ada)
    models.append(bagging)
    models.append(ran_forest)
    models.append(extra_tree)
    models.append(gradient_boosting)
    
    return models, tokenizer

# collect out of fold predictions form k-fold cross validation
def get_out_of_fold_predictions(X, y, models, tokenizer):
	meta_X, meta_y = list(), list()
	# define split of data
	kfold = StratifiedKFold(n_splits=5, shuffle=True)
	# enumerate splits
	for train_ix, test_ix in kfold.split(X, y):
		fold_yhats = list()
		train_X, test_X = X[train_ix], X[test_ix]
		train_y, test_y = y[train_ix], y[test_ix]
		meta_y.extend(test_y)
		# fit and make predictions with each sub-model
		for i, model in enumerate(models):
			if i == 0:
				train_input = bert_encode(train_X, tokenizer, max_len=160)
				early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', verbose=1, patience=10, mode='max', restore_best_weights=True)
				model.fit(train_input, train_y, validation_split=0.25, epochs=10,batch_size=16,  callbacks=[early_stopping])
				_test_X = bert_encode(test_X, tokenizer, max_len=160)
				yhat = model.predict(_test_X)
				_y = []
				for __y in yhat:
					one_prob = __y[0]
					zero_prob = 1 - one_prob
					_y.append([zero_prob, one_prob])
				yhat = _y
			else:
				model.fit(train_X, train_y)
				yhat = model.predict_proba(test_X)
			# store columns
			fold_yhats.append(yhat)
		# store fold yhats as columns
		meta_X.append(hstack(fold_yhats))
	return vstack(meta_X), asarray(meta_y)

# fit all base models on the training dataset
def fit_base_models(X, y, models, bert_tok):
	for i,model in enumerate(models):
		if i == 0:
			train_input = bert_encode(X, bert_tok, max_len=160)
			early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', verbose=1, patience=10, mode='max', restore_best_weights=True)
			model.fit(train_input, y, validation_split=0.10, epochs=10,batch_size=16,  callbacks=[early_stopping])
		else:  
			model.fit(X, y)

# fit a meta model
def fit_meta_model(X, y):
	model = ExtraTreesClassifier(n_estimators=30)
	model.fit(X, y)
	return model


# make predictions with stacked model
def super_learner_predictions(X, models, meta_model, bert_tok):
	meta_X = list()
	for i, model in enumerate(models):
		if i == 0:
			x = bert_encode(X, bert_tok, max_len=160)
			yhat = model.predict(x)
			_y = []
			for y in yhat:
				one_prob = y[0]
				zero_prob = 1 - one_prob
				_y.append([zero_prob, one_prob])
			yhat =_y
		else:
			yhat = model.predict_proba(X)
		meta_X.append(yhat)
	meta_X = hstack(meta_X)
	# predict
	return meta_model.predict(meta_X)


"""
test_df, res = read_test(test_path="/kaggle/input/nlp-getting-started/test.csv")
X, y = split_data(df)
X_test = split_data(test_df, _t=False)

# get models
models, bert_tok = get_models()
# get out of fold predictions
meta_X, meta_y = get_out_of_fold_predictions(X, y, models,bert_tok)
print('Meta ', meta_X.shape, meta_y.shape)
# fit base models
fit_base_models(X, y, models, bert_tok)
# fit the meta model
meta_model = fit_meta_model(meta_X, meta_y)

# evaluate meta model
yhat = super_learner_predictions(X_test, models, meta_model, bert_tok)

print("YHat: {}".format(yhat))
dump_preds(res, yhat, out="submission_ensemble.csv")
"""


# # Embedding Stacking!
# Unfortunately the GPU allowed in Kaggle is not enough to keep multiple embeddings, if Elmo is one of them. So this is just an example code, witouth an official score.

# In[ ]:


get_ipython().system('pip install flair')
from flair.data import Corpus
from flair.datasets import TREC_6
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings, StackedEmbeddings, BertEmbeddings, ELMoEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.data import Sentence
import pandas as pd
from keras.layers import Input, Dense, GRU, Bidirectional, Flatten
from keras.optimizers import Adam
from keras.models import Model
import numpy as np


def generateTrainingData(dataset, batch_size, max_length, num_classes, emb_size,
                        stacked_embedding):
  
  x_batch = []
  y_batch = []
  while True:
    data = dataset.sample(frac=1)
    for index, row in data.iterrows():
 
        my_sent = row["text"]
        sentence = Sentence(my_sent)
        stacked_embedding.embed(sentence)
        
        x = []
        for token in sentence:
          x.append(token.embedding.cpu().detach().numpy())
          if len(x) == max_length:
            break
        
        while len(x) < max_length:
          x.append(np.zeros(emb_size))
        
        y = np.zeros(num_classes)
        y[row["target"]] = 1
        
        x_batch.append(x)            
        y_batch.append(y)

        if len(y_batch) == batch_size:
          yield np.array(x_batch), np.array(y_batch)

          x_batch = []
          y_batch = []

def generatePredictionData(dataset, batch_size, max_length, num_classes, emb_size, stacked_embedding):
  
  x_batch = []
  while True:
    for text in dataset['text'].values:
 
        my_sent = text
        sentence = Sentence(my_sent)
        stacked_embedding.embed(sentence)
        
        x = []
        for token in sentence:
          x.append(token.embedding.cpu().detach().numpy())
          if len(x) == max_length:
            break
        
        while len(x) < max_length:
          x.append(np.zeros(emb_size))
          
        x_batch.append(x)            
        if len(x_batch) == batch_size:
          yield np.array(x_batch)

          x_batch = []

def get_stacked_embeddings():
  stacked_embedding = StackedEmbeddings([ELMoEmbeddings(), WordEmbeddings("en")])
  print("Stacked embedding size: {}".format(stacked_embedding.embedding_length))
  embedding_size = stacked_embedding.embedding_length
  return stacked_embedding, embedding_size

def declare_model(batch_size, max_len, emb_size, gru_size, num_classes):
 
  sample = Input(batch_shape=(batch_size, max_len, emb_size))
  gru_out = Bidirectional(GRU(gru_size, return_sequences=True))(sample)
  gru_out = Flatten()(gru_out)
  predictions = Dense(num_classes, activation='sigmoid')(gru_out)

  model = Model(inputs=sample, outputs=[predictions])
  model.compile(optimizer=Adam(),loss='binary_crossentropy', metrics=["acc"])
  print(model.summary())

  return model

"""
BATCH_SIZE = 256
MAX_LEN = 150
GRU_SIZE = 20
NUM_CLASSES=2 
EPOCHS = 1

stacked_embedding, embedding_length = get_stacked_embeddings()

m = declare_model(batch_size=BATCH_SIZE, max_len=MAX_LEN, emb_size=embedding_length, gru_size=GRU_SIZE, num_classes=NUM_CLASSES)


gen = generateTrainingData(df, batch_size=BATCH_SIZE, max_length=MAX_LEN, num_classes=NUM_CLASSES, emb_size=embedding_length,
  stacked_embedding= stacked_embedding)
print(gen)
steps_per_epoch = len(df)/BATCH_SIZE
m.fit_generator(gen, steps_per_epoch=1, epochs=EPOCHS, workers=1)

df_test = df[:10]
test_gen = list(generatePredictionData(df_test,  batch_size=BATCH_SIZE, max_length=MAX_LEN, num_classes=NUM_CLASSES, 
        emb_size=embedding_length, stacked_embedding=stacked_embedding))

print(np.argmax(m.predict_generator(test_gen, steps=1), axis=1))
"""

