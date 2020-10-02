#!/usr/bin/env python
# coding: utf-8

# Using the tensorflow keras highlevel api, build a model that can process a question and piece of text jointly and return a 1-word answer.
# Work in progress, if you know how to finish it please share.

# In[ ]:


import tensorflow as tf
layers = tf.keras.layers
Input = tf.keras.Input


# In[ ]:


text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500


# In[ ]:


text_input = Input(shape=(None,), dtype='int32', name='text')
text_input


# In[ ]:


embedded_text = layers.Embedding(64, text_vocabulary_size)(text_input)
embedded_text


# In[ ]:


encoded_text = layers.LSTM(32)(embedded_text)
encoded_text


# In[ ]:


question_input = Input(shape=(None,), dtype='int32', name='question')
question_input


# In[ ]:


embedded_question = layers.Embedding(32, question_vocabulary_size)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)


# In[ ]:


concatenated = layers.concatenate([encoded_text, encoded_question],
                                 axis=-1)


# In[ ]:


answer = layers.Dense(answer_vocabulary_size,
                      activation='softmax')(concatenated)


# In[ ]:


model = tf.keras.Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
             metrics=['acc'])
model.summary()


# In[ ]:


'''
# pydotplus is not available in gpu kernels
from keras.utils import plot_model 
plot_model(model, to_file='../input/multi-input-model.png')
'''
from IPython.display import Image
Image(filename='../input/multi-input-model.png') 


# ## Train

# In[ ]:


import numpy as np


# In[ ]:


num_samples = 1000
max_length = 100


# In[ ]:


text = np.random.randint(1,
                        text_vocabulary_size,
                        size=(num_samples, max_length))


# In[ ]:


question = np.random.randint(1,
                            question_vocabulary_size,
                            size=(num_samples, max_length))
answers = np.random.randint(0,
                           1,
                           size=(num_samples, answer_vocabulary_size))


# In[ ]:


# fitting using a list of inputs
# model.fit([text, question], answers, epochs=10, batch_size=128)


# In[ ]:


# fitting using a dictionary of inputs (only if inputs are named)
model.fit({'text': text, 'question': question},
          answers, epochs=10, batch_size=128)


# In[ ]:


model


# ## Inference

# In[ ]:


text = "A federal judge in Texas is weighing a request by 20 states to suspend the Affordable Care Act (ACA), a move that could lead to chaos in the health insurance market, some industry experts worry."
question = "Who answered the request?"


# In[ ]:


predict_input = np.array([1,2,3])
predict_input.shape


# In[ ]:


model.input_shape


# In[ ]:


prediction = model.predict([predict_input, predict_input])
prediction


# In[ ]:


sum(prediction[0])

