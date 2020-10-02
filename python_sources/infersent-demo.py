#!/usr/bin/env python
# coding: utf-8

# ## InferSent demo
# #### Disclaimer: this notebook is adaptation of [InferSent Github repo](https://github.com/facebookresearch/InferSent)
# InferSent is a sentence embeddings method that provides semantic representations for English sentences. It is trained on natural language inference data and generalizes well to many different tasks.
# 
# In this notebook we only present InferSent pretrained on GloVe. For model pretrained on fastText, please fork this notebook and substitute appropriate binaries from official repository

# ## Environment Setup

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from random import randint
import numpy as np
import torch
import shutil
import string
import nltk.data
import matplotlib

matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)


# In[ ]:


# here we need to restructure working directory, so that script imports working properly
shutil.copytree("/kaggle/input/infersent/", "/kaggle/working/infersent")
get_ipython().system(' mv /kaggle/working/infersent/* /kaggle/working/')


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# TODO: add encoder to dataset as well\n# If this cell freezes, probably you haven't enabled Internet access for the notebook\n! mkdir encoder\n! curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl")


# ## Load Model

# In[ ]:


model_version = 1
MODEL_PATH = "encoder/infersent%s.pkl" % model_version
W2V_PATH = '/kaggle/input/glove-840b-300d/glove.840B.300d.txt'
VOCAB_SIZE = 1e5  # Load embeddings of VOCAB_SIZE most frequent words
USE_CUDA = False  # Keep it on CPU if False, otherwise will put it on GPU


# In[ ]:


from models import InferSent
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model = model.cuda() if USE_CUDA else model\n\nmodel.set_w2v_path(W2V_PATH)\n\nmodel.build_vocab_k_words(K=VOCAB_SIZE)')


# ## Encode Sentences

# We need to specify input sentences as a list, where each punctuation sign is padded with a space from both sides

# In[ ]:


sentences = ['Everyone really likes the newest benefits',
 'The Government Executive articles housed on the website are not able to be searched .',
 'I like him for the most part , but would still enjoy seeing someone beat him .',
 'My favorite restaurants are always at least a hundred miles away from my house .',
 'What a day !',
 'What color is it ?',
 'I know exactly .']


# If you have a text in a single string, you can use `format_text` helper funtion to split it by sentences and pad appropriately

# In[ ]:


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def format_text(text):
    global tokenizer
    padded_text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    return tokenizer.tokenize(padded_text)

text = 'Everyone really likes the newest benefits. The Government Executive articles housed on the website are not able to be searched.''I like him for the most part, but would still enjoy seeing someone beat him. My favorite restaurants are always at least a hundred ''miles away from my house. What a day! What color is it? I know exactly.'

sentences = format_text(text)
sentences


# * gpu mode : >> 1000 sentences/s
# * cpu mode : ~100 sentences/s

# In[ ]:


embeddings = model.encode(sentences, bsize=128, tokenize=False, verbose=True)
print('nb sentences encoded : {0}'.format(len(embeddings)))


# ## Visualization

# In[ ]:


np.linalg.norm(model.encode(['the cat eats.']))


# In[ ]:


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

cosine(model.encode(['the cat eats.'])[0], model.encode(['the cat drinks.'])[0])


# In[ ]:


idx = randint(0, len(sentences) - 1)
_, _ = model.visualize(sentences[idx])


# In[ ]:


_, _ = model.visualize('The cat is drinking milk.')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model.build_vocab_k_words(5e5) # getting 500K words vocab\n_, _ = model.visualize("barack-obama is the former president of the United-States.")')

