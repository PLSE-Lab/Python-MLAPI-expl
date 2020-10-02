#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import sklearn
import tensorflow as tf 
import seaborn as sb


# In[ ]:


csvfile = '../input/sms-spam-collection-dataset/spam.csv'
import chardet
with open(csvfile, 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large
    print(result)
df = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv', encoding='Windows-1252')
df


# In[ ]:


df.rename(columns={"v1": "target", "v2": "text"}, inplace=True)
df['target'] = (df['target']== 'spam').astype(int)


# In[ ]:


sb.distplot(df.text.str.len())
# df.text.str.len().hist()
# Most texts have a length less than 200


# In[ ]:


get_ipython().run_cell_magic('time', '', '!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py\nimport tensorflow_hub as hub \nimport tokenization\nmodule_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"\n# module_url = \'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2\'\nbert_layer = hub.KerasLayer(module_url, trainable=True)\n\nvocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()\ndo_lower_case = bert_layer.resolved_object.do_lower_case.numpy()\ntokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)')


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

def build_model(bert_layer, max_len=512):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    net = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(32, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(net)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# In[ ]:


import sklearn.model_selection

X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(df.text.values, df.target, test_size=0.1, random_state=0)
X_train = bert_encode(X_train, tokenizer, max_len=200)
X_val = bert_encode(X_val, tokenizer, max_len=200)


# In[ ]:


model = build_model(bert_layer, max_len=200)
model.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', "checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)\nearlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)\n\ntrain_history = model.fit(\n    X_train, y_train,\n    validation_split=0.05,\n    epochs=30,\n    callbacks=[checkpoint, earlystopping],\n    batch_size=16,\n    verbose=1\n)")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'import sklearn.metrics \nmodel.load_weights(\'model.h5\')\ny_preds = model.predict(X_val)\ny_preds = (y_preds >= 0.5).astype(int)\nprint("Validation accuracy score", sklearn.metrics.accuracy_score(y_val, y_preds))')

