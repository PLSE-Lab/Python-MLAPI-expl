#!/usr/bin/env python
# coding: utf-8

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
Dropout_num = 0


# In[ ]:


test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")


# In[ ]:


def load_training(training_path="/kaggle/input/nlp-getting-started/train.csv"):
    df = pd.read_csv(training_path)
    
    print(df.head(10))
    return df

df = load_training()


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


# In[ ]:


ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]
df.loc[df['id'].isin(ids_with_target_error),'target'] = 0
df[df['id'].isin(ids_with_target_error)]


# # Utils for models

# In[ ]:


def read_test(test_path="/kaggle/input/nlp-getting-started/test.csv"):
    
    my_df = pd.read_csv(test_path)
    
    res_df = my_df[['id']]
    my_df = my_df[['text']]
    
    
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

    


# # BERT TfHub

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
    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# In[ ]:


get_ipython().run_cell_magic('time', '', 'module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"\nbert_layer = hub.KerasLayer(module_url, trainable=True)')


# In[ ]:



vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

train_input = bert_encode(df.text.values, tokenizer, max_len=160)
test_input = bert_encode(test_df.text.values, tokenizer, max_len=160)
train_labels = df.target.values

model = build_model(bert_layer, max_len=160)
model.summary()



# In[ ]:


train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=3,
    batch_size=16
)

model.save('model.h5')

 

