#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# We will use the official tokenization script created by the Google team
get_ipython().system('wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py')


# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
import tokenization


# In[ ]:


train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")


# In[ ]:


train.head()


# In[ ]:


text=train['text']


# In[ ]:


text.head()


# In[ ]:


text=text.str.lower()


# In[ ]:


import string


# In[ ]:


def remove_punctuation(text):
    return text.translate(str.maketrans('','',string.punctuation))
text_clean=text.apply(lambda text:remove_punctuation(text))


# In[ ]:


text_clean.head()


# In[ ]:


def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)


# In[ ]:


import re
text_clean=text_clean.apply(lambda x : remove_URL(x))


# In[ ]:


def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)


# In[ ]:


text_clean=text_clean.apply(lambda x : remove_html(x))


# In[ ]:


df = pd.DataFrame({"text": text_clean})
df.head()


# In[ ]:


train.update(df)
train.head()
train.drop(columns=['keyword','location','id'],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x=train['text']
y=train['target']


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.4,random_state=42)


# In[ ]:


module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)


# In[ ]:


vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()


# In[ ]:


do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()


# In[ ]:


tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)


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


train_input = bert_encode(xtrain.values, tokenizer, max_len=160)


# In[ ]:


train_labels = ytrain.values


# In[ ]:


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


# In[ ]:


test_input = bert_encode(xtest.values, tokenizer, max_len=160)


# In[ ]:


test_labels = ytest.values


# In[ ]:


ypred = model.predict(test_input)


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


result=mean_squared_error(ypred,ytest)


# In[ ]:


result


# bert code is taken from xhlulu's notebook and i modified it to fit my data.
