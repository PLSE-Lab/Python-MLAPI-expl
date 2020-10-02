#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py as tokenization')


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf
tf.executing_eagerly ()
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub

import tokenization 


# In[ ]:


tf.executing_eagerly ()


# In[ ]:


train=pd.read_csv('../input/nlp-getting-started/train.csv')
test=pd.read_csv('../input/nlp-getting-started/test.csv')
submission=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')


# In[ ]:


def clean_text(text):
    text = re.sub(r'https?://\S+', '', text) # Remove link
    text = re.sub(r'\n',' ', text) # Remove line breaks
    text = re.sub('\s+', ' ', text).strip() # Remove leading, trailing, and extra spaces
    return text
def find_hashtags(tweet):
    return " ".join([match.group(0)[1:] for match in re.finditer(r"#\w+", tweet)]) or 'no'

def find_mentions(tweet):
    return " ".join([match.group(0)[1:] for match in re.finditer(r"@\w+", tweet)]) or 'no'

def find_links(tweet):
    return " ".join([match.group(0)[:] for match in re.finditer(r"https?://\S+", tweet)]) or 'no'

def process_text(df):
    
    df['text_clean'] = df['text'].apply(lambda x: clean_text(x))
    df['hashtags'] = df['text'].apply(lambda x: find_hashtags(x))
    df['mentions'] = df['text'].apply(lambda x: find_mentions(x))
    df['links'] = df['text'].apply(lambda x: find_links(x))
    # df['hashtags'].fillna(value='no', inplace=True)
    # df['mentions'].fillna(value='no', inplace=True)
    
    return df
    
train = process_text(train)
test= process_text(test)


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
tokenizer = tokenization.FullTokenizer(vocab_file,do_lower_case)


# In[ ]:


train_input = bert_encode(train.text_clean.values,tokenizer, max_len=160)
test_input = bert_encode(test.text_clean.values,tokenizer, max_len=160)
train_labels = train.target.values


# In[ ]:


model = build_model(bert_layer, max_len=160)
model.summary()


# In[ ]:


train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=3,
    batch_size=8
)


# In[ ]:


fig,axes=plt.subplots(ncols=2,figsize=(10,3),dpi=100)
sns.lineplot(x=train_history.epoch,y=train_history.history['accuracy'],ax=axes[0],label='Train',color='blue')
sns.lineplot(x=train_history.epoch,y=train_history.history['loss'],ax=axes[1],label='Train',color='red')
axes[0].set_title('TF Hub Bert Accuracy')
axes[1].set_title('TF Hub Bert Loss')

plt.show()


# In[ ]:


train_pred = model.predict(train_input)
train_pred_int = train_pred.round().astype('int')


# In[ ]:


print(classification_report(train['target'].values,train_pred_int))


# In[ ]:


confusion_matrix=metrics.confusion_matrix(train['target'].values,train_pred_int)
sns.heatmap(confusion_matrix,annot=True)
plt.title('Confusion Matrix \n TF Hub Bert',fontsize=14)
plt.show()
sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])
print('Sensitivity :% .2f' % sensitivity)
specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
print('Specificity :% .2f '% specificity)


# In[ ]:


test_pred = model.predict(test_input)
submission['target']=test_pred.round().astype('int')
submission.to_csv('mysubmission.csv',index=False)

