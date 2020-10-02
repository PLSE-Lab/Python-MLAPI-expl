#!/usr/bin/env python
# coding: utf-8

# ## Word embeddings lead to scientific discovery

# In this notebook, the abstracts in the CORD-19 corpus are used to build word embeddings with *Word2Vec*. Distances in the high-dimensional space mapped by *Word2Vec* reveal bits of knowledge around COVID-19. Examples include the **autonomous discovery** of the similarity between COVID-19 and SARS and MERS, of the radical difference between COVID-19 and seasonal flu, and of state-of-the-art trial therapies. The model developed here can become a valuable tool to explore the literature related to COVID-19.

# This work was largely inspired by a recent [article](https://www.nature.com/articles/s41586-019-1335-8) on how scientific discovery (in that case of high-performance materials) is often latent in past literature and how Natural Language Processing can track hidden relationship that lead to invention. I, and other Kaggle users (see [this](https://www.kaggle.com/tarunpaparaju/covid-19-dataset-gaining-actionable-insights) or [this](https://www.kaggle.com/tylersuard/mat2vec-covid-papers-unexpected-word-asociations?scriptVersionId=30790274) notebooks, for example), thought that the same principle can be applied to the CORD-19 database.

# ## Method

# [This](https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial) notebook is a great hands-on introduction to word embeddings and I have borrowed pieces of code from it.

# ## Reading the Abstracts

# In[ ]:


import re
import pandas as pd
from collections import defaultdict
import spacy
import en_core_web_sm
import numpy as np
import matplotlib.pyplot as plt


# Abstracts are usually **less noisy** than the whole texts, because they are more concise and not redundant. This is why they are preferred in this analysis.

# In[ ]:


metadata=pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
metadata=metadata[metadata["abstract"]==metadata["abstract"]]
print(metadata.shape)


# ## Cleaning the Abstracts

# In[ ]:


nlp = en_core_web_sm.load(disable=['ner', 'parser']) # disabling Named Entity Recognition for speed

def cleaning(doc):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.lemma_ for token in doc if not token.is_stop]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt) > 2:
        return ' '.join(txt)
brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower().replace("abstract", "") for row in metadata['abstract'])
txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]
df_clean = pd.DataFrame({'clean': txt})
df_clean = df_clean.dropna().drop_duplicates()


# Setting up bigrams:

# In[ ]:


from gensim.models.phrases import Phrases, Phraser
sent = [row.split() for row in df_clean['clean']]
phrases = Phrases(sent, min_count=30, progress_per=10000)
bigram = Phraser(phrases)
sentences = bigram[sent]


# Some checks:

# In[ ]:


word_freq = defaultdict(int)
for sent in sentences:
    for i in sent:
        word_freq[i] += 1


# In[ ]:


MIN_COUNT=6
DICT=[key for key in word_freq.keys() if word_freq[key]>MIN_COUNT]


# In[ ]:


"covid" in DICT


# ## Training the model

# In[ ]:


import multiprocessing

from gensim.models import Word2Vec

cores = multiprocessing.cpu_count()

w2v_model = Word2Vec(min_count=MIN_COUNT,
                     window=2,
                     size=200,
                     sample=1e-4, 
                     alpha=0.01,
                     min_alpha=0.0001, 
                     negative=15,
                     workers=cores-1)

w2v_model.build_vocab(sentences, progress_per=10000)

w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

w2v_model.init_sims(replace=True)


# ## Exploring the model

# Let's see if the model is reasonable. First of all, let's look at similarities:

# In[ ]:


w2v_model.wv.most_similar("treatment",topn=10)


# Ok, this makes sense; most of the words related to "treatment" are **synonyms**.

# In[ ]:


w2v_model.wv.most_similar(positive=["china"],topn=10)


# Wow, my model has autonomously learned **Geography**; it knows that Taiwan is a country, as well as China, and that the provinces Hubei and Guangdong, and the cities Wuhan and Shenzhen are all in China.

# In[ ]:


w2v_model.wv.most_similar(positive=["oseltamivir"],topn=10)


# Great, the model seems to know that the antiviral **Oseltamivir** (commercial name **Tamiflu**) is a **Neuraminidase Inhibitor** (**NAI**), similar to **Zanamivir**, and that is given to patients in combination with antibiotics (**Vancomycin**, **Erythromycin**, **Fluoroquinolone**, **Clarithromycin**, ...). I didn't know that!

# In[ ]:


w2v_model.wv.most_similar(positive=["malaria"],topn=10)


# It also "knows" that malaria is caused by the Protozoa *Plasmodium Falciparum* and *Plasmodium Vivax*.

# ## Fighting COVID-19

# Most related words to "COVID":

# In[ ]:


w2v_model.wv.most_similar(positive=["covid"],topn=10) #ncp=novel coronavirus pneumonia


# In the [R&D Blueprint Report](https://apps.who.int/iris/bitstream/handle/10665/330680/WHO-HEO-RDBlueprint%28nCoV%29-2020.1-eng.pdf?ua=1) compiled by WHO on January 24th, 2020, it is suggested that the antiviral **Remdesivir** is the most promising candidate to treat COVID-19. If I look in the "neighbourhood" of this drug:

# In[ ]:


w2v_model.wv.most_similar(positive=["remdesivir"],topn=20) #recommended by WHO on Jan 24


# I find potential alternative candidates to fight COVID-19:
# - **Nitazoxanide** is a broad-spectrum antiviral [[source]](https://en.wikipedia.org/wiki/Nitazoxanide).
# - **Lopinavir**, **Ritonavir**, and **Darunavir** are used against HIV. On March, 17th 2020 the Italian Agency for Pharmaceuticals (AIFA) has allowed experimental treatment of COVID-19 with a combination of these two drugs [[source]](https://www.aifa.gov.it/-/azioni-intraprese-per-favorire-la-ricerca-e-l-accesso-ai-nuovi-farmaci-per-il-trattamento-del-covid-19).
# - **Favipiravir** is used against flu. On March, 23rd 2020 AIFA has started to consider experimental therapies with this drug in the most-hit regions in Italy [[source]](https://www.aifa.gov.it/-/favipiravir-aggiornamento-della-valutazione-della-cts), despite the very limited scientific evidence of its effectiveness.
# - **Sofosbuvir** is used against Hepatitis C. It has been suggested against COVID-19 [[source]](https://www.biorxiv.org/content/10.1101/2020.01.30.927574v1.full.pdf), but the scientific evidence is extremely limited.
# - **Umifenovir** (commercial name: Arbidol or Abidol) is also being studied against COVID-19, but AIFA has stated that there is no scientific evidence of it being effective against COVID-19 [[source]](https://www.aifa.gov.it/-/aifa-precisa-uso-umifenovir-su-covid-19-non-autorizzato-in-europa-e-usa-scarse-evidenze-scientifiche-sull-efficacia).
# - **Artesunate**, **Amodiaquine**, **Aminoquinaline**, and **Hydroxychloroquine** are antimalaria drugs. The latter has been approved for testing against COVID-19 on March, 17th 2020 by (AIFA) [[source]](https://www.aifa.gov.it/-/azioni-intraprese-per-favorire-la-ricerca-e-l-accesso-ai-nuovi-farmaci-per-il-trattamento-del-covid-19), and mentioned by Donald Trump [[source]](https://twitter.com/realDonaldTrump/status/1242120391054757900) and Elon Musk [[source]](https://twitter.com/elonmusk/status/1239776019856461824) on Twitter (please mind that neither of them is a medical doctor though!)
# 

# ## COVID-19 is NOT a simple flu

# Given this list of diseases:

# In[ ]:


diseases=["covid","coronavirus","sars","mers","covid_pneumonia","novel_coronavirus","ncov","cov","coronavirus_sars","sars_cov","influenza","seasonal_influenza","h_n","swine_flu","avian_flu"]


# I have applied Principal Component Analysis to the 200-dimensional space mapped by *Word2Vec* to downfold it to 2D and plotted the components of each disease in this low-dimensional representation:

# In[ ]:


X = w2v_model[diseases].T
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
result = pca.fit_transform(X.T)
plt.figure(figsize=(15,5))
plt.scatter(result.T[0],result.T[1])
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("Component 1",fontsize=14)
plt.ylabel("Component 2",fontsize=14)
for i in range(0,len(diseases)):
    plt.annotate(diseases[i],xy=(result.T[0][i],result.T[1][i]), xycoords='data',
            xytext=(-15, 5), textcoords='offset points', fontsize=15)


# The plot nicely shows that **two** separate **clusters** form: one contains very bad respiratory illnesses (COVID, SARS, MERS), and the other less severe types of flu. Therefore, it can be concluded that COVID-19 is **NOT A COMMON FLU** and must not be underestimated.

# ## Conclusions

# Common knowledge related to COVID-19 stems out naturally from word embeddings produced from the CORD-19 database. Potential effective therapies being experimented are close in terms of distance in the high-dimensional space produced by Word2Vec, thus eventual new treatments can arise by exploring the **neighbourhood** of the therapies already in use.

# Word embeddings also show unequivocally that the literature does not regard COVID-19 as a seasonal flu, suggesting that the **containment** precautions taken in many countries are largely **supported** by scientific evidence.

# *To the patients, doctors and nurses all over the world: keep on fighting.*

# In[ ]:




