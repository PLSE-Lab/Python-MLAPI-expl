#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
import tensorflow_hub as hub
import keras
import keras.backend as K
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import *
from keras import Model

import pickle    
import os

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
                
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')
test = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')
submission = pd.read_csv('/kaggle/input/google-quest-challenge/sample_submission.csv')


# In[ ]:


module_url = "/kaggle/input/universalsentenceencoderlarge4/"
embed = hub.load(module_url)


# In[ ]:


# For the keras Lambda
def UniversalEmbedding(x):
    results = embed(tf.squeeze(tf.cast(x, tf.string)))["outputs"]
    print(results)
    return keras.backend.concatenate([results])


# In[ ]:


# setup training data
targets = [
        'question_asker_intent_understanding',
        'question_body_critical',
        'question_conversational',
        'question_expect_short_answer',
        'question_fact_seeking',
        'question_has_commonly_accepted_answer',
        'question_interestingness_others',
        'question_interestingness_self',
        'question_multi_intent',
        'question_not_really_a_question',
        'question_opinion_seeking',
        'question_type_choice',
        'question_type_compare',
        'question_type_consequence',
        'question_type_definition',
        'question_type_entity',
        'question_type_instructions',
        'question_type_procedure',
        'question_type_reason_explanation',
        'question_type_spelling',
        'question_well_written',
        'answer_helpful',
        'answer_level_of_information',
        'answer_plausible',
        'answer_relevance',
        'answer_satisfaction',
        'answer_type_instructions',
        'answer_type_procedure',
        'answer_type_reason_explanation',
        'answer_well_written'    
    ]

input_columns = ['question_title','question_body','answer']

X1 = train[input_columns[0]].values.tolist()
X2 = train[input_columns[1]].values.tolist()
X3 = train[input_columns[2]].values.tolist()
X1 = [x.replace('?','.').replace('!','.') for x in X1]
X2 = [x.replace('?','.').replace('!','.') for x in X2]
X3 = [x.replace('?','.').replace('!','.') for x in X3]

X = [X1,X2,X3]
y = train[targets].values.tolist()


# In[ ]:


# build network
def swish(x):
    return K.sigmoid(x) * x

embed_size = 512 #must be 512 for univerasl embedding layer

input_text1 = Input(shape=(1,), dtype=tf.string)
embedding1 = Lambda(UniversalEmbedding, output_shape=(embed_size,))(input_text1)
input_text2 = Input(shape=(1,), dtype=tf.string)
embedding2 = Lambda(UniversalEmbedding, output_shape=(embed_size,))(input_text2)
input_text3 = Input(shape=(1,), dtype=tf.string)
embedding3 = Lambda(UniversalEmbedding, output_shape=(embed_size,))(input_text3)

x = Concatenate()([embedding1,embedding2,embedding3])
x = Dense(256, activation=swish)(x)
x = Dropout(0.4)(x)
x = BatchNormalization()(x)
x = Dense(64, activation=swish, kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = Dropout(0.4)(x)
x = BatchNormalization()(x)
output = Dense(len(targets),activation='sigmoid',name='output')(x)


# In[ ]:


model = Model(inputs=[input_text1,input_text2,input_text3], outputs=[output])
model.summary()


# In[ ]:


# clean up as much as possible
import gc
print(gc.collect())


# In[ ]:


# Train the network
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=2, min_lr=1e-7, verbose=1)
optimizer = Adadelta()

model.compile(optimizer=optimizer, loss='binary_crossentropy')
model.fit(X, [y], epochs=20, validation_split=.1,batch_size=32,callbacks=[reduce_lr])


# In[ ]:


# prep test data
X1 = test[input_columns[0]].values.tolist()
X2 = test[input_columns[1]].values.tolist()
X3 = test[input_columns[2]].values.tolist()
X1 = [x.replace('?','.').replace('!','.') for x in X1]
X2 = [x.replace('?','.').replace('!','.') for x in X2]
X3 = [x.replace('?','.').replace('!','.') for x in X3]

pred_X = [X1,X2,X3]


# In[ ]:


# Make a prediction
pred_y = model.predict(pred_X)


# In[ ]:


# Check the submission
submission = pd.read_csv('/kaggle/input/google-quest-challenge/sample_submission.csv')
submission[targets] = pred_y
submission.head()


# In[ ]:


# Save the result
submission.to_csv("submission.csv", index = False)

