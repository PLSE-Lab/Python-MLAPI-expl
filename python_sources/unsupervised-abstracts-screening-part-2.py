#!/usr/bin/env python
# coding: utf-8

# ## Import packages

# In[ ]:


import pandas as pd
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy.stats import loguniform, uniform, randint

RANDOM_STATE = 1563


# ## Introduction
# 
# **<span style="color:red">The notebook is based upon our previous work. They should be evaluated together!</span>** [Link](https://www.kaggle.com/quittend/unsupervised-abstracts-screening-part-1)

# ## Read data

# In[ ]:


df = pd.read_csv('/kaggle/input/cleaning-cord-19-metadata/cord_metadata_cleaned.csv')
print(f'There are {len(df)} studies.')


# ## BioWordVec + GMM

# ### Extract features
# I couldn't extract features using BioWordVec in the kernel due to the out of memory issues, thus I did it on my personal computer using the following snippet. I decided to "max pool features over time" as suggested in many papers like [Rethinking Complex Neural Network Architectures forDocument Classification](https://cs.uwaterloo.ca/~jimmylin/publications/Adhikari_etal_NAACL2019.pdf).
# 
# ```
# from gensim.models.keyedvectors import KeyedVectors
# 
# model = KeyedVectors.load_word2vec_format(
#     fname='./BioWordVec_PubMed_MIMICIII_d200.vec.bin', 
#     binary=True
# )
# 
# nlp = spacy.load('en_core_sci_sm')
# 
# def vectorize(text: str):
#     features = np.array([
#         model[token.text] 
#         for token in nlp(text)
#         if token.text in model.vocab
#     ])
#     
#     
#     return features.max(axis=0) if features.size != 0 else np.zeros(200)
#     
# df['text_vector'] = df['text'].apply(vectorize)
# ```

# In[ ]:


embeddings = np.load('/kaggle/input/biowordvec-precomputed-cord19/biowordvec.npy')
print(f'Embedding matrix has shape: {embeddings.shape}')


# ### Define model

# In[ ]:


estimator = GaussianMixture(
    n_components=10,
    covariance_type='full', 
    max_iter=100, 
    n_init=1, 
    init_params='kmeans', 
    random_state=RANDOM_STATE, 
)


# ### Hyperparameter search
# A single set of hyperparameters for 4 splits on 4 cores with one KMeans initialization for GMM takes roughly 3.5 min including refiting model on the whole dataset.

# In[ ]:


N_ITER = 20
N_SPLITS = 4

param_distributions = {
    "n_components": randint(2, 256),
    "covariance_type": ['diag', 'full', 'spherical'],
}

cv = KFold(
    n_splits=N_SPLITS, 
    shuffle=True, 
    random_state=RANDOM_STATE
)

hp_search = RandomizedSearchCV(
    estimator=estimator,
    param_distributions=param_distributions,
    n_iter=N_ITER,
    n_jobs=N_SPLITS,
    cv=cv,
    verbose=1,
    random_state=RANDOM_STATE,
    return_train_score=True,
    refit=True
)

hp_search.fit(embeddings)
best_model = hp_search.best_estimator_


# In[ ]:


print(f'Best validation likelihood: {hp_search.best_score_}')


# In[ ]:


print(f'Best params: {hp_search.best_params_}')


# ## Visualize data
# 

# In[ ]:


df['cluster'] = best_model.predict(embeddings)


# ### How many elements does each cluster have?

# In[ ]:


cluster_count = df['cluster'].value_counts().sort_values()

ax = cluster_count.plot(kind='bar', figsize=(15, 5))
ax.set_xticks([])
ax.set_xlabel("Cluster id")
ax.set_ylabel("Count")
ax.grid(True)


# Clusters seem to be very well balanced!

# ## Save results
# Such results are not easily interpretable as TF-IDF, but there are much less clusters, meaning that such approach is worth evaluation.

# In[ ]:


(df
    .drop(columns=['title_lang', 'abstract_lang', 'distance'])
    .to_csv('/kaggle/working/cord_metadata_word2vec.csv', index=False)
)

