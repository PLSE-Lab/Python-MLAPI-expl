#!/usr/bin/env python
# coding: utf-8

# # Dynamic Bernoulli Embeddings
# 
# Dynamic Bernoulli Embeddings, discussed [here](http://www.cs.columbia.edu/~blei/papers/RudolphBlei2018.pdf), are a way to train word embeddings that smoothly change with time. Documents are grouped into time buckets, and every word in the vocabulary gets an embedding for each timestep. These embeddings smoothly drift over time, so embeddings across different timesteps can be meaningfully compared with cosine or euclidean distance.
# 
# After finding the paper authors' code a bit challenging to use, I created my own PyTorch based implementation. This notebook shows how to apply it to the UN General Debate corpus. You can find my code [here](https://github.com/llefebure/dynamic_bernoulli_embeddings).

# In[ ]:


get_ipython().system('pip install git+git://github.com/llefebure/dynamic_bernoulli_embeddings.git')


# In[ ]:


import pickle
import re

import numpy as np
import pandas as pd
from dynamic_bernoulli_embeddings.analysis import DynamicEmbeddingAnalysis
from dynamic_bernoulli_embeddings.training import train_model
from nltk import word_tokenize as nltk_word_tokenize
from gensim.corpora import Dictionary
from tqdm.notebook import tqdm
tqdm.pandas()


# ## Preprocessing
# 
# The `train_model` function expects a pandas data frame with at least two columns, `bow` and `time`. `bow` is just a list of words in the document, and `time` is expected to be an integer in `[0, T)` where `T` is the total number of timesteps. It also expects a dictionary mapping tokens to their index in `[0, V)` where `V` is the size of the vocabulary. Any token found in the dataset but not in the vocabulary is ignored.
# 
# Here, I do some light preprocessing of the text and use `gensim` to build the dictionary. Documents are bucketed by year.

# In[ ]:


def _bad_word(word):
    if len(word) < 2:
        return True
    if any(c.isdigit() for c in word):
        return True
    if "/" in word:
        return True
    return False

def word_tokenize(text):
    text = re.sub(r"co-operation", "cooperation", text)
    text = re.sub(r"-", " ", text)
    words = [w.lower().strip("'.") for w in nltk_word_tokenize(text)]
    words = [w for w in words if not _bad_word(w)]
    return words


# In[ ]:


dataset = pd.read_csv("../input/un-general-debates/un-general-debates.csv")
dataset["bow"] = dataset.text.progress_apply(word_tokenize)
dataset["time"] = dataset.year - dataset.year.min()


# In[ ]:


dictionary = Dictionary(dataset.bow)
dictionary.filter_extremes(no_below=10, no_above=1.)
dictionary.compactify()
print(len(dictionary))


# ## Training
# 
# With the inputs prepared, we can now train the model. We'll set aside 10% of the dataset for validation, run for 6 epochs, and use an embedding dimension of 100.

# In[ ]:


model, loss_history = train_model(
    dataset, dictionary.token2id, validation=.1, num_epochs=6, k=100)


# In[ ]:


loss_history.loss.plot(title="Training Loss")


# In[ ]:


loss_history.l_pos.plot(title="Positive")


# In[ ]:


loss_history.l_neg.plot(title="Negative")


# In[ ]:


loss_history.l_prior.plot(title="Prior")


# In[ ]:


np.save("embeddings", model.get_embeddings())
loss_history.to_csv("loss_history.csv", index=False)
pickle.dump(dictionary.token2id, open("dictionary.pkl", "wb"))


# ## Analysis
# 
# We can use the trained embeddings to do some cool analysis such as:
# * finding which embeddings changed the most over time (absolute drift)
# * looking up the "neighborhood" of an embedding and seeing how this has changed over time
# * looking for change points where an embedding significantly changed from one timestep to the next indicating some significant event

# In[ ]:


emb = DynamicEmbeddingAnalysis(model.get_embeddings(), dictionary.token2id)


# In[ ]:


emb.absolute_drift()


# In[ ]:


over_time = {}
for i in range(0, dataset.time.max() + 1, 5):
    col = str(dataset.year.min() + i)
    over_time[col] = emb.neighborhood("climate", i, 10)
pd.DataFrame(over_time)


# In[ ]:


over_time = {}
for i in range(0, dataset.time.max() + 1, 5):
    col = str(dataset.year.min() + i)
    over_time[col] = emb.neighborhood("afghanistan", i, 10)
pd.DataFrame(over_time)


# In[ ]:


pd.DataFrame([(dataset.year.min() + i, term) for i, term in emb.change_points(20)], columns=["Year", "Term"])

