#!/usr/bin/env python
# coding: utf-8

# This notebook is a quick example of tensorflow hub. To set up this notebook, I:
# 
# * Installed tensorflow-hub using pip install
# * Enabled Internet so that we can access the modules stored on-line
# 
# This will allow this sample code to run:

# In[ ]:


import tensorflow as tf
import tensorflow_hub as hub

with tf.Graph().as_default():
  module_url = "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1"
  embed = hub.Module(module_url)
  embeddings = embed(["A long sentence.", "single-word",
                      "http://example.com"])

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    print(sess.run(embeddings))


# To run TensorFlow Hub in your own notebook, you can either follow the steps above or fork & use this notebook. :)
