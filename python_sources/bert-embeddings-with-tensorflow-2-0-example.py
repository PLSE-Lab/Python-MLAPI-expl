#!/usr/bin/env python
# coding: utf-8

# # BERT Embeddings with TensorFlow 2.0
# With the new release of TensorFlow, this Notebook aims to show a simple use of the BERT model.
# - See BERT on paper: https://arxiv.org/pdf/1810.04805.pdf
# - See BERT on GitHub: https://github.com/google-research/bert
# - See BERT on TensorHub: https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1
# - See 'old' use of BERT for comparison: https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb

# ## Update TF
# We need Tensorflow 2.0 and TensorHub 0.7 for this Notebook

# If TensorFlow Hub is not 0.7 yet on release, use dev.
# 
# **Disclaimer**: bert-tensorflow is not the latest version of BERT. To use bert with TF 2.0, you should use tensorflow/models where the model is updated to tf2.0. To resolve this issue, I'll use `tf.gfile = tf.io.gfile` at one point of the code. If you use Google Colab, it uses the latest bert version (no need for `pip install bert-tensorflow`), but I couldn't reproduce the same in Kaggle
# 
# If you know, how to install module from here tf/models, please share in comment!
# The latest BERT at tf/models: https://github.com/tensorflow/models/tree/master/official/nlp/bert

# In[ ]:


get_ipython().system('pip install --upgrade tensorflow')
get_ipython().system('pip install bert-tensorflow')
get_ipython().system('pip install tf-hub-nightly')


# In[ ]:


import tensorflow as tf
import tensorflow_hub as hub
print("TF version: ", tf.__version__)
print("Hub version: ", hub.__version__)


# ## Import modules

# In[ ]:


import tensorflow_hub as hub
import tensorflow as tf
from bert.tokenization import FullTokenizer     # Still from bert module
from tensorflow.keras.models import Model       # Keras is the new high level API for TensorFlow
import math


# Building model using tf.keras and hub. from sentences to embeddings.
# 
# Inputs:
#  - input token ids (tokenizer converts tokens using vocab file)
#  - input masks (1 for useful tokens, 0 for padding)
#  - segment ids (for 2 text training: 0 for the first one, 1 for the second one)
# 
# Outputs:
#  - pooled_output of shape `[batch_size, 768]` with representations for the entire input sequences
#  - sequence_output of shape `[batch_size, max_seq_length, 768]` with representations for each input token (in context)

# In[ ]:


max_seq_length = 128  # Your choice here.
input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=True)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])


# In[ ]:


model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])


# Generating segments and masks based on the original BERT

# In[ ]:


# See BERT paper: https://arxiv.org/pdf/1810.04805.pdf
# And BERT implementation convert_single_example() at https://github.com/google-research/bert/blob/master/run_classifier.py

def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids


# In[ ]:


# Google Colab don't need this. FullTokenizer is not updated to tf2.0 yet
tf.gfile = tf.io.gfile


# Import tokenizer using the original vocab file

# In[ ]:


vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)


# ## Test BERT embedding generator model

# In[ ]:


s = "This is a nice sentence."
stokens = tokenizer.tokenize(s)

input_ids = get_ids(stokens, tokenizer, max_seq_length)
input_masks = get_masks(stokens, max_seq_length)
input_segments = get_segments(stokens, max_seq_length)

print(stokens)
print(input_ids)
print(input_masks)
print(input_segments)


# Generate Embeddings using the pretrained model

# In[ ]:


pool_embs, all_embs = model.predict([[input_ids],[input_masks],[input_segments]])


# ## Pooled embedding vs [CLS] as sentence-level representation
# 
# Previously, the [CLS] token's embedding were used as sentence-level representation (see the original paper). However, here a pooled embedding were introduced. This part is a short comparison of the two embedding using cosine similarity

# In[ ]:


def square_rooted(x):
    return math.sqrt(sum([a*a for a in x]))


def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return numerator/float(denominator)


# In[ ]:


cosine_similarity(pool_embs[0], all_embs[0][0])


# In[ ]:




