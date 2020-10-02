#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This notebook is a pure fork of the great notebook by @xhlulu : https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub (version 7)
# except training data (train.csv)
# 
# V2: epochs = 2
# 
# V3: epochs = 4
# 
# V4: epochs = 10 (monitor = 'val_accuracy')

# In[ ]:


import numpy as np
import pandas as pd
train_orig = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
train_orig.shape


# While studying this model and my own models, I discovered that these kind of predictions are so sensitive to the training data. Next, I read tweets in training data and figure out, that some of them have errors:

# In[ ]:


ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]
train_orig[train_orig['id'].isin(ids_with_target_error)]


# Let's fix these tweets records:

# In[ ]:


train_orig.at[train_orig['id'].isin(ids_with_target_error),'target'] = 0
train_orig[train_orig['id'].isin(ids_with_target_error)]


# After doing this, luckily (or not) score could be slightly higher. But for sure our model is slightly better! Imho this could be a hint about: how to improve your score/model. There is much more work about this training data as far as I can see :) 

# ### My second shot was to take tweets from another source. Luckily there is a kaggle dataset containing 1.6m tweets that can be found here: https://www.kaggle.com/kazanova/sentiment140
# 
# Although some of them are tweeted around 10 years ago, I choose randomly **1000** of them, and only **16 **were classified as **real disaster**. However, after reading these tweets I find that only two of those are about disaster! After corrections, our additional training data looks like this:

# In[ ]:


train_add = pd.read_csv("/kaggle/input/real-or-not-nlp-with-disaster-tweets-addings/train_add.csv")
train_add


# It's an important and hard lesson for our classifier. Let's append it to the original training data set:

# In[ ]:


train = train_orig.append(train_add)
train.shape


#  Good luck!

# # About this kernel
# 
# I've seen a lot of people pooling the output of BERT, then add some Dense layers. I also saw various learning rates for fine-tuning. In this kernel, I wanted to try some ideas that were used in the original paper that did not appear in many public kernel. Here are some examples:
# * *No pooling, directly use the CLS embedding*. The original paper uses the output embedding for the `[CLS]` token when it is finetuning for classification tasks, such as sentiment analysis. Since the `[CLS]` token is the first token in our sequence, we simply take the first slice of the 2nd dimension from our tensor of shape `(batch_size, max_len, hidden_dim)`, which result in a tensor of shape `(batch_size, hidden_dim)`.
# * *No Dense layer*. Simply add a sigmoid output directly to the last layer of BERT, rather than experimenting with different intermediate layers.
# * *Fixed learning rate, batch size, epochs, optimizer*. As specified by the paper, the optimizer used is Adam, with a learning rate between 2e-5 and 5e-5. Furthermore, they train the model for 3 epochs with a batch size of 32. I wanted to see how well it would perform with those default values.
# 
# I also wanted to share this kernel as a **concise, reusable, and functional example of how to build a workflow around the TF2 version of BERT**. Indeed, it takes less than **50 lines of code to write a string-to-tokens preprocessing function and model builder**.
# 
# ## References
# 
# * Source for `bert_encode` function: https://www.kaggle.com/user123454321/bert-starter-inference
# * All pre-trained BERT models from Tensorflow Hub: https://tfhub.dev/s?q=bert

# In[ ]:


# We will use the official tokenization script created by the Google team
get_ipython().system('wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py')


# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub

import tokenization


# # Helper Functions

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


# # Load and Preprocess
# 
# - Load BERT from the Tensorflow Hub
# - Load CSV files containing training data
# - Load tokenizer from the bert layer
# - Encode the text into tokens, masks, and segment flags

# In[ ]:


get_ipython().run_cell_magic('time', '', 'module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"\nbert_layer = hub.KerasLayer(module_url, trainable=True)')


# In[ ]:


train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")


# In[ ]:


vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)


# In[ ]:


train_input = bert_encode(train.text.values, tokenizer, max_len=160)
test_input = bert_encode(test.text.values, tokenizer, max_len=160)
train_labels = train.target.values


# # Model: Build, Train, Predict, Submit

# In[ ]:


model = build_model(bert_layer, max_len=160)
model.summary()


# In[ ]:


checkpoint = ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True)

train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=10,
    callbacks=[checkpoint],
    batch_size=16
)


# In[ ]:


model.load_weights('model.h5')
test_pred = model.predict(test_input)


# In[ ]:


submission['target'] = test_pred.round().astype(int)
submission.to_csv('submission.csv', index=False)

