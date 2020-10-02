#!/usr/bin/env python
# coding: utf-8

# # About this kernel
# 
# This kernel shows you how to use load the GPT-2 weights in Tensorflow v1, and run them prewritten text.

# In[ ]:


get_ipython().system('cp -r ../input/openai-gpt2-weights/src/. .')
get_ipython().system('pip install tensorflow-gpu==1.14 --quiet')


# In[ ]:


import json
import os
import numpy as np
import tensorflow as tf

import encoder, model, sample


# # Helper Function

# In[ ]:


def run_model_on_prompt(model_name, models_dir, prompt, 
                        top_k=40, top_p=0.9, length=None, seed=2019,
                        batch_size=1, n_samples=0, temperature=1):
    """
    This function is mostly copied from this file: 
    https://github.com/openai/gpt-2/blob/master/src/interactive_conditional_samples.py
    
    It is slighthly modified for simplicity
    """
    
    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    
    context_tokens = enc.encode(prompt)
    
    if length is None:
        length = hparams.n_ctx // 2

    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)

        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        out = sess.run(output, feed_dict={
            context: [context_tokens]
        })[:, len(context_tokens):]
    
    return enc.decode(np.squeeze(out))


# # Reddit r/WritingPrompts

# The prompt:

# In[ ]:


raw_text = "The year is 1910. Adolf Hitler, a struggling artist, has fought off dozens of assasination attemps by well meaning time travelers, but this one is different. This traveller doesn't want to kill Hitler, he wants to teach him to paint. He pulls off his hood to reveal the frizzy afro of Bob Ross."
print(raw_text)


# ## Small (124M)

# In[ ]:


get_ipython().run_cell_magic('time', '', "out_small = run_model_on_prompt(\n    models_dir='/kaggle/input/openai-gpt2-weights/124M', model_name='124M', prompt=raw_text\n)")


# In[ ]:


print(out_small)


# ## Medium (355M)

# In[ ]:


get_ipython().run_cell_magic('time', '', "out_medium = run_model_on_prompt(models_dir='/kaggle/input/openai-gpt2-weights/355M', model_name='355M', prompt=raw_text)")


# In[ ]:


print(out_medium)


# ## Large (774M)

# In[ ]:


get_ipython().run_cell_magic('time', '', "out_large = run_model_on_prompt(models_dir='/kaggle/input/openai-gpt2-weights/774M', model_name='774M', prompt=raw_text)")


# In[ ]:


print(out_large)


# ## Largest (1.5B)

# In[ ]:


get_ipython().run_cell_magic('time', '', "out_largest = run_model_on_prompt(models_dir='/kaggle/input/openai-gpt2-weights/1558M', model_name='1558M', prompt=raw_text)")


# In[ ]:


print(out_largest)


# # Completing Obama's speech

# Taken from this archive: http://www.washingtonpost.com/wp-dyn/articles/A19751-2004Jul27.html

# In[ ]:


speech = """OBAMA: Thank you so much. Thank you.

(APPLAUSE)

Thank you. Thank you so much. Thank you so much.

(APPLAUSE)

Thank you, Dick Durbin. You make us all proud.

On behalf of the great state of Illinois...

(APPLAUSE)

... crossroads of a nation, land of Lincoln, let me express my deep gratitude for the privilege of addressing this convention. Tonight is a particular honor for me because, let's face it, my presence on this stage is pretty unlikely.

My father was a foreign student, born and raised in a small village in Kenya. He grew up herding goats, went to school in a tin- roof shack. His father, my grandfather, was a cook, a domestic servant to the British.

OBAMA: But my grandfather had larger dreams for his son. Through hard work and perseverance my father got a scholarship to study in a magical place, America, that's shown as a beacon of freedom and opportunity to so many who had come before him.

(APPLAUSE)

While studying here my father met my mother. She was born in a town on the other side of the world, in Kansas."""
print(speech)


# ## Small (124M)

# In[ ]:


get_ipython().run_cell_magic('time', '', "out_small = run_model_on_prompt(\n    models_dir='/kaggle/input/openai-gpt2-weights/124M', model_name='124M', prompt=speech\n)")


# In[ ]:


print(out_small)


# ## Medium (355M)

# In[ ]:


get_ipython().run_cell_magic('time', '', "out_medium = run_model_on_prompt(\n    models_dir='/kaggle/input/openai-gpt2-weights/355M', model_name='355M', prompt=speech\n)")


# In[ ]:


print(out_medium)


# ## Large (774M)

# In[ ]:


get_ipython().run_cell_magic('time', '', "out_large = run_model_on_prompt(\n    models_dir='/kaggle/input/openai-gpt2-weights/774M', model_name='774M', prompt=speech\n)")


# In[ ]:


print(out_large)


# ## Largest (1.5B)

# In[ ]:


get_ipython().run_cell_magic('time', '', "out_largest = run_model_on_prompt(\n    models_dir='/kaggle/input/openai-gpt2-weights/1558M', model_name='1558M', prompt=speech\n)")


# In[ ]:


print(out_largest)

