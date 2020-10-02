#!/usr/bin/env python
# coding: utf-8

# I'm going to generate topic models on a corpora using gensim and spacy.
# Topic modelling aims to genrate discover different sources of document theme, a.k.a topics. In topic modeling we suppose each document is a combination of multiple topics. Topic Models helps to find out how much a document is related to a topic. Automatically extracting key topics in a collection of articles, emails, discussion texts, etc. Topic Modeling is also useful in many nlp tasks, including information retrieval, classification and summarization. 
# 
# Let's do this with reuters news dataset. This dataset contains ~10K documents and 1.3 Million words.

# In[ ]:


import nltk
nltk.download('reuters')
from nltk.corpus import reuters
reuters_corpora = [reuters.raw(fid) for fid in reuters.fileids()]


# I'm going to use spacy to process texts in this corpora. Processing texts is more straigh-forward in spacy.
# In order to use pos tags, you should download a spacy model file. Make sure you have "en_core_web_sm" installed on your machine. To do so, run below command:

# In[ ]:


# !python -m spacy download en_core_web_sm


# In[ ]:


import spacy
from spacy.symbols import DET, X, NUM, PRON
from tqdm import tqdm
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore

nlp = spacy.load('en_core_web_sm')


# In LDA model, each document is supposed as a bag of different words. Each topic is a probability distribution on words and Each document is a probabilistic combinations of topics. Suppose we want to generate a Document of lenght N (this is an old-school algorithm and each document is just a bag-of-words, forget about the order! this is not deep learning). Anyway, for each N word we choose a topic (with different probabilities) and use this topic to generate a word. Simple and Easy :)
# LDA algorithm aims to estimate these probablitlies.
# 

# Lets start with tokenizing each text and removing non-important words. I'm going to use word lemmas instead of words. This improves the model output. I also drop some words, including stop-words, numbers, determiners and pronouns.

# In[ ]:


processed_corpora = [
    [token.lemma_ for token in nlp(doc_text) \
        if not(token.is_punct or token.is_stop or token.is_space or token.pos in [DET, NUM, PRON])]
    for doc_text in tqdm(reuters_corpora)]


# Now, lets prepare this data to feed gensim LDA model.

# In[ ]:


# Create Dictionary
gensim_dict = corpora.Dictionary(processed_corpora)

# gensim_dict.token2id = reuters_vocab
processed_corpora = [gensim_dict.doc2bow(text) for text in processed_corpora]


# In[ ]:


lda_model = LdaMulticore(processed_corpora,
                        id2word=gensim_dict,
                        num_topics=10,
                        workers= 2)
lda_model


# In[ ]:


lda_model.print_topics()


# let's save this model to disk first

# In[ ]:


from gensim.test.utils import datapath

model_file = datapath("gensim_model")
lda_model.save(model_file)

# You can load it using:
# from gensim.models.ldamulticore import LdaModel
# lda_model = LdaModel.load(model_file)


# **OPTIONAL:** you can visalize topics using pyLDAvis module

# In[ ]:


import pyLDAvis
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, processed_corpora, gensim_dict)
vis


# Now, we can add this model as an attribute to spacy Doc objects. Just add an extension attribute to Doc Objects and you have access to 5 key topic of each document:

# In[ ]:


from spacy.tokens import Doc
topics = lambda doc: lda_model[gensim_dict.doc2bow([token.lemma_ for token in doc])][0]
Doc.set_extension('topics', getter=topics)


# In[ ]:


doc = nlp(u'The decisive factor now is the behavior of the U.S. president, who basically told the crown prince, we are giving you free rein as long as you buy enough weapons and other things from us')
print(doc._.topics)


# In[ ]:




