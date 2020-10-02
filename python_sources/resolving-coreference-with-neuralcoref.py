#!/usr/bin/env python
# coding: utf-8

# # Resolving coreference with neuralcoref

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm_notebook


# There are few out-of-the-box libraries that support or specifically built for coreference resolution. Most wide-known are [CoreNLP](https://stanfordnlp.github.io/CoreNLP/coref.html), [Apache OpenNLP](https://opennlp.apache.org/) and [neuralcoref](https://github.com/huggingface/neuralcoref). In this short notebook, we will explore neuralcoref 3.0, a coreference resolution library by Huggingface.
# 
# First, let's install neuralcoref 3.0. To do this, we need to slightly downgrade spacy (neuralcoref is not compatible with the new cymem version used by the current version of spacy).

# In[ ]:


MODEL_URL = "https://github.com/huggingface/neuralcoref-models/releases/"             "download/en_coref_md-3.0.0/en_coref_md-3.0.0.tar.gz"


# In[ ]:


get_ipython().system('pip install spacy==2.0.12')


# In[ ]:


get_ipython().system('pip install {MODEL_URL}')


# In[ ]:


get_ipython().system('python -m spacy download en_core_web_md')


# ## A small neuralcoref tutorial
# 
# How does this lib work? Let's find out!
# 
# First,we need to load the model:

# In[ ]:


import en_coref_md

nlp = en_coref_md.load()


# Now we need a sentence with coreference. 
# 
# A boring theoretical reminder: coreference happens when* two different words denote the same entity* in the real world. In this competition, we deal with pronomial coreference. It comes in two flavors:
# 1. *Anaphora*, when a pronoun follows a noun: "John looked at me. He was clearly angry".
# 2. *Cataphora*, when it is vice versa: "When she opened the door, Jane realized that it was cold outside"
# 
# Let's start with two simple sentences with two anaphoric coreferences:

# In[ ]:


test_sent = "The doctor came in. She held a paper in her hand."


# Using neuralcoref is not really different from using plain spacy.

# In[ ]:


doc = nlp(test_sent)


# To check if any kind of coreference was detected, `has_coref` attribute of the extension (referred to as `_`) is used:

# In[ ]:


doc._.has_coref


# Great! We found something, let's see what exactly:

# In[ ]:


doc._.coref_clusters


# You can go to the [website](https://huggingface.co/coref/?text=The%20doctor%20came%20in.%20She%20held%20a%20paper%20in%20her%20hand.) and play with the tool. It outputs cool resolution graphs like this one:
# 
# ![graph](http://i66.tinypic.com/wtbmdi.png)

# You can get the entity and coreferring pronouns from these clusters by simple indexing. The objects returned are in fact ordinary spacy `span`s.

# In[ ]:


doc._.coref_clusters[0].main


# In[ ]:


doc._.coref_clusters[0].mentions


# ## Deciding which entity the pronoun refers to
# 
# In competition data, the position of the entities and the pronoun comes as an offset from the beginning. Let's write a small function that will resolve coreference in a string and decide whether any of detected coreferring entities correspond to given offsets.
# 

# In[ ]:


def is_inside(offset, span):
    return offset >= span[0] and offset <= span[1]

def is_a_mention_of(sent, pron_offset, entity_offset_a, entity_offset_b):
    doc = nlp(sent)
    if doc._.has_coref:
        for cluster in doc._.coref_clusters:
            main = cluster.main
            main_span = main.start_char, main.end_char
            mentions_spans = [(m.start_char, m.end_char) for m in cluster.mentions                               if (m.start_char, m.end_char) != main_span]
            if is_inside(entity_offset_a, main_span) and                     np.any([is_inside(pron_offset, s) for s in mentions_spans]):
                return "A"
            elif is_inside(entity_offset_b, main_span) and                     np.any([is_inside(pron_offset, s) for s in mentions_spans]):
                return "B"
            else:
                return "NEITHER"
    else:
        return "NEITHER"


# A small test:

# In[ ]:


# "The doctor came in. She held a paper in her hand."
entity_offset_a = test_sent.index("doctor")
entity_offset_b = test_sent.index("paper")
pron_offset = test_sent.index("She")

is_a_mention_of(test_sent, pron_offset, entity_offset_a, entity_offset_b)


# ## Testing on the dataset 

# In[ ]:


gap_train = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv", 
                       delimiter='\t', index_col="ID")


# In[ ]:


gap_train.head()


# In[ ]:


def predict(df):
    pred = pd.DataFrame(index=df.index, columns=["A", "B", "NEITHER"]).fillna(False)
    for i, row in tqdm_notebook(df.iterrows()):
        pred.at[i, is_a_mention_of(row["Text"], row["Pronoun-offset"], row["A-offset"], row["B-offset"])] = True
    return pred


# In[ ]:


train_preds = predict(gap_train)


# In[ ]:


gap_train["NEITHER"] = np.logical_and(~gap_train["A-coref"], ~gap_train["B-coref"])


# In[ ]:


gap_train[["A-coref", "B-coref", "NEITHER"]].describe()


# In[ ]:


train_preds.describe()


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(gap_train[["A-coref", "B-coref", "NEITHER"]], train_preds[["A", "B", "NEITHER"]]))


# We can see that though precision is quite good, we have very low recall. What can be done?
# 1. Remove excessive sentenes: if entities and the pronoun are contained in two sentences, we can strip other sentences.
# 2. Use neuralcoref's verdicts as a feature for another classifier (we would have to transform verdicts into probabilities anyway).
