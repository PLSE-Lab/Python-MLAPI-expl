#!/usr/bin/env python
# coding: utf-8

# The audiences and subjects columns in the news data are two of the more difficult features to work with. This notebook aims to explore these features, primarily by using embeddings to see how the terms relate to one another.

# In[ ]:


import numpy as np
np.random.seed(13)
from matplotlib import pyplot as plt
import itertools
import scipy
import sklearn.decomposition
import sklearn.manifold
import json
import gc


# In[ ]:


EMBEDDING_SIZE = 8
MIN_OCCURRENCES = 10


# ## Load Data

# In[ ]:


from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()


# In[ ]:


(market_data, news_data) = env.get_training_data()


# ### Convert the audience and subject features to tuples
# 
# The audiences and subjects featurs are strings, to make the processing easier we will convert them to tuples

# In[ ]:


get_ipython().run_cell_magic('time', '', 'news_data[\'subjects_tuples\'] = news_data[\'subjects\'].copy()\nsubjects_cats = news_data[\'subjects_tuples\'].cat.categories\nsubjects_cats = [eval(c.replace("{", "(").replace("}", ")")) for c in subjects_cats]\nnews_data[\'subjects_tuples\'].cat.categories = subjects_cats\n\ndel subjects_cats')


# In[ ]:


news_data[['assetCodes', 'time', 'subjects', 'subjects_tuples']].head(3)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'news_data[\'audiences_tuples\'] = news_data[\'audiences\'].copy()\naudiences_cats = news_data[\'audiences_tuples\'].cat.categories\naudiences_cats = [eval(c.replace("{", "(").replace("}", ")")) for c in audiences_cats]\nnews_data[\'audiences_tuples\'].cat.categories = audiences_cats\n\ndel audiences_cats')


# In[ ]:


news_data[['assetCodes', 'time', 'audiences', 'audiences_tuples']].head(3)


# ## Create Embeddings
# 
# To create the embedding, first we will create a multi-hot encoded array, and then use dimensionality reduction to obtain vectors of our desired size (EMBEDDING_SIZE). To perform the multi-hot encoding we will use the following function.

# In[ ]:


def pd_categorical_to_dummies(series, min_occurrences=0):
    
    features_cats_evals = series.cat.categories
    unique_features = list(set(itertools.chain(*features_cats_evals)))
    
    num_unique_features = len(unique_features)
    
    features_map = {k:v for v, k in enumerate(unique_features)}
    features_cats_factorized = [[features_map[k] for k in l] for l in features_cats_evals]
    
    features_lengths = [
        len(features_cats_factorized[i]) 
        for i in series.cat.codes
    ]
    
    features_cats_rows = np.arange(series.shape[0]).repeat(features_lengths)
    
    features_cats_cols = np.array([
        v for c in 
        series.cat.codes
        for v in features_cats_factorized[c]
    ])
    
    total_length = len(features_cats_cols)
    
    dummies = scipy.sparse.coo_matrix(
        (np.ones(total_length, dtype=np.bool), (features_cats_rows, features_cats_cols)),
        shape=(series.shape[0], num_unique_features),
        dtype=np.bool
    )
    
    dummies = dummies.tocsr()
    
    m = dummies.sum(axis=0).A[0] > min_occurrences
    
    dummies = dummies[:, m]
    unique_features = [a for a, mm in zip(unique_features, m) if mm == 1]
    
    return dummies, unique_features


# This is another helper function we will use later

# In[ ]:


def get_similar(w, embeddings, features, max_features=10):
    
    i = features.index(w)
    v = embeddings[i]
    similarities = (embeddings @ v)
    
    ii = np.argsort(similarities)[::-1]
    
    similarities = similarities[ii[:max_features + 1]]
    
    similar_words = [features[j] for j in ii[:max_features + 1]]
        
    m = similarities > 1 - 1e-6
        
    assert w in np.array(similar_words)[m]
    
    similar_words.remove(w)
    
    return similar_words


# ## Subject Embedding
# 
# Lets start by creating an embedding for the subjects

# In[ ]:


get_ipython().run_cell_magic('time', '', "subject_dummies, subjects = pd_categorical_to_dummies(\n    news_data['subjects_tuples'],\n    MIN_OCCURRENCES\n)")


# In[ ]:


subject_dummies.shape, len(subjects)


# In[ ]:


svd_reducer = sklearn.decomposition.TruncatedSVD(
    n_components=EMBEDDING_SIZE,
    algorithm='randomized',
    n_iter=5,
    random_state=None,
    tol=0.0
)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'svd_reducer.fit(subject_dummies.T)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '_subject_embeddings = svd_reducer.transform(subject_dummies.T)')


# In[ ]:


assert np.abs(_subject_embeddings.sum(axis=1)).min() != 0


# Normalise the vectors

# In[ ]:


subject_embeddings = _subject_embeddings/np.linalg.norm(_subject_embeddings, axis=1)[:, np.newaxis]
# subject_embeddings[np.isnan(subject_embeddings)] = 0


# In[ ]:


assert np.isnan(subject_embeddings).sum() == 0


# ### Visualise
# 
# Now we can use the vectors to see how subjects relate to one another. The first thing we will do is use TSNE to visualise the embedding.
# 
# There are two sources on the internet that describe what some of the subjects relate to:
# 
#  [s3.amazonaws.com/tr-liaison-documents/public/Reuters_News_Topics_External.xls](http://s3.amazonaws.com/tr-liaison-documents/public/Reuters_News_Topics_External.xls)
# 
#  https://liaison.reuters.com/tools/topic-codes
#  
#  Of the two the first, the spreadsheet, appears to be the more complete list.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'subject_tsne = sklearn.manifold.TSNE(n_components=2).fit_transform(subject_embeddings)\nprint(subject_tsne.shape)')


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.scatter(subject_tsne[:, 0], subject_tsne[:, 1], s=1)

for i, txt in enumerate(subjects):
    if i % 20 == 0:
        ax.annotate(txt, (subject_tsne[i, 0], subject_tsne[i, 1]), fontsize=10)

plt.show()


# Next we can use cosine similarity to find which subjects are similar to a given subject

# The FUND subject is related to other banking and fund subjects

# In[ ]:


" ".join(get_similar("FUND", subject_embeddings, subjects))


#     FUND 	Funds
#     BANK 	Banks (TRBC)
#     PVE 	 Private Equity Funds
#     BSVC 	Banking Services (TRBC)

# The GOLF subject is related to other sports and more general news

# In[ ]:


" ".join(get_similar("GOLF", subject_embeddings, subjects))


#     GOLF 	Golf
#     PREV     Previews / Schedules / Diaries
#     GEN      General News
#     ODD 	 Human Interest / Brights / Odd News
#     ICEH 	Ice Hockey

# The EPMICS subject is related to diseases and health issues

# In[ ]:


" ".join(get_similar("EPMICS", subject_embeddings, subjects))


#     EPMICS   Epidemics
#     INFDIS   Infectious Diseases
#     WOMHEA   Women's Health
#     COMDIS   Communicable Diseases

# The TWAVE subject is related to natural disasters

# In[ ]:


" ".join(get_similar("TWAVE", subject_embeddings, subjects))


#     TWAVE 	Tsunami
#     WLDPWS    Wind Power? (WINPWR = Wind Farms)
#     QUAK 	 Earthquakes
#     VIO 	  Civil Unrest
#     TRD       International Trade

# The subjects embedding appears to have done a reasonable job at grouping similar terms together (based on a few examples).
# 
# Lets save the results.

# In[ ]:


with open("subjects.json", "w") as f:
    json.dump(subjects, f)
np.save("subject_embeddings.npy", subject_embeddings)


# In[ ]:


if False:
    del subject_dummies, subjects, subject_embeddings
del subject_tsne, _subject_embeddings, svd_reducer
gc.collect()


# ### Audience Embedding
# 
# Next we can perform the same steps on the audience data

# In[ ]:


get_ipython().run_cell_magic('time', '', "audience_dummies, audiences = pd_categorical_to_dummies(\n    news_data['audiences_tuples'], \n    MIN_OCCURRENCES\n)")


# In[ ]:


audience_dummies.shape, len(audiences)


# In[ ]:


svd_reducer = sklearn.decomposition.TruncatedSVD(
    n_components=EMBEDDING_SIZE,
    algorithm='randomized',
    n_iter=5,
    random_state=None,
    tol=0.0
)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'svd_reducer.fit(audience_dummies.T)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '_audience_embeddings = svd_reducer.transform(audience_dummies.T)')


# In[ ]:


assert np.abs(_audience_embeddings.sum(axis=1)).min() != 0


# Normalise the vectors

# In[ ]:


audience_embeddings = _audience_embeddings/np.linalg.norm(_audience_embeddings, axis=1)[:, np.newaxis]
# audience_embeddings[np.isnan(audience_embeddings)] = 0


# In[ ]:


assert np.isnan(audience_embeddings).sum() == 0


# ### Visualise
# 
# As with the subjects embedding we can use TSNE to visualise the vectors.
# 
# Unlike for the subjects, I have been unable to find any resources to describe what each of the audiences mean, which makes analysing the results harder.

# In[ ]:


audience_tsne = sklearn.manifold.TSNE(n_components=2).fit_transform(audience_embeddings)
print(audience_tsne.shape)


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.scatter(audience_tsne[:, 0], audience_tsne[:, 1], s=1)

for i, txt in enumerate(audiences):
    if i % 5 == 0:
        ax.annotate(txt, (audience_tsne[i, 0], audience_tsne[i, 1]))

plt.show()


# Again, lets have a look at some similar terms for a selection of audiences

# In[ ]:


" ".join(get_similar("OIL", audience_embeddings, audiences))


# In[ ]:


" ".join(get_similar("MTL", audience_embeddings, audiences))


#     MTL    Metal
#     OIL    Oil

# In[ ]:


" ".join(get_similar("NZP", audience_embeddings, audiences))


# In[ ]:


" ".join(get_similar("FN", audience_embeddings, audiences))


# It is more difficult to tell how good the vectors are at grouping similar audiences compared with the subjects due to the lack of any information on what the abbreviations mean. However, looking at the results I would have expected to be able to see some more obvious patterns.
# 
# Lets save the audience embedding and see if we can use the subjects, which we have some confidence in, to get an understand of what some of the audience abbreviations mean.

# In[ ]:


with open("audiences.json", "w") as f:
    json.dump(audiences, f)
np.save("audience_embeddings.npy", audience_embeddings)


# In[ ]:


if False:
    del audience_dummies, audiences
del audience_embeddings, audience_tsne, _audience_embeddings, svd_reducer
gc.collect()


# ## Audience Embedding using Subject Map
# 
# The embedding learnt for the audiences was not particularly encouraging. Since we have some confidence about the quality of the subjects data lets take a look at which subjects best describe each audience.

# In[ ]:


audience_subject_map = {}
num = 5

global_subject_proportions = subject_dummies.sum(axis=0).A[0]/subject_dummies.shape[0]

for i in range(len(audiences)):
    m = audience_dummies[:, i].A[:, 0]
    a = audiences[i]
    
    c = subject_dummies[m].sum(axis=0).A[0]
    p = c/subject_dummies[m].shape[0]
    # s = np.abs(p - global_subject_proportions)
    s = np.clip(p - global_subject_proportions, 0, np.inf)
    
    ii = np.argsort(s)[::-1][:num]
    
    subs = np.array(subjects)[ii].tolist()
    cnts = c[ii]
    
    # print(a, subs)
    
    audience_subject_map[a] = subs
    
    #break


# In[ ]:


audience_subject_map['OIL']


# The OIL audience appears to be related to energy subjects:
# 
#     COM    Commodities
#     NRG    Energy Markets
#     ENR    Energy (Legacy)

# In[ ]:


audience_subject_map['MTL'] # Metal??


# It is less clear what the MTL audience is related to but assuming it is to do with metal the commodities subject is of note:
#   
#     COM    Commodities    
#     BLR    Content produced in Bangalore
#     FIN    Financials (Legacy)

# In[ ]:


audience_subject_map['FN'] # Finland ??


# The FN audience seems to be related to European countries:
# 
#     FI    Finland
#     NORD  Nordic States

# In[ ]:


audience_subject_map['NZP'] # New Zealand ??


# The NZP appear to be related to New Zealand

# There appears to be a strong pattern in which subjects best describe each audience, so let try and use the subject vectors to create an audience vector.

# In[ ]:


# subject_embeddings = np.load("subject_embeddings.npy")


# In[ ]:


_audience_embeddings_using_sub = []

for i in range(len(audiences)):
    a = audiences[i]
    
    subs = audience_subject_map[a]
    
    ii = [i for i, s in enumerate(subjects) if s in subs]
    
    e = subject_embeddings[ii].mean(axis=0)
    
    _audience_embeddings_using_sub.append(e)
    
_audience_embeddings_using_sub = np.array(_audience_embeddings_using_sub)

print(_audience_embeddings_using_sub.shape)


# In[ ]:


audience_embeddings_using_sub = _audience_embeddings_using_sub/np.linalg.norm(_audience_embeddings_using_sub, axis=1)[:, np.newaxis]


# In[ ]:


audience_using_sub_tsne = sklearn.manifold.TSNE(n_components=2).fit_transform(
    audience_embeddings_using_sub
)
print(audience_using_sub_tsne.shape)


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.scatter(audience_using_sub_tsne[:, 0], audience_using_sub_tsne[:, 1], s=1)

for i, txt in enumerate(audiences):
    if i % 2 == 0:
        ax.annotate(txt, (audience_using_sub_tsne[i, 0], audience_using_sub_tsne[i, 1]))

plt.show()


# In[ ]:


" ".join(get_similar("OIL", audience_embeddings_using_sub, audiences))


# In[ ]:


" ".join(get_similar("MTL", audience_embeddings_using_sub, audiences))


# In[ ]:


" ".join(get_similar("FN", audience_embeddings_using_sub, audiences))


# In[ ]:


" ".join(get_similar("NZP", audience_embeddings_using_sub, audiences))


# Its still hard to tell if the audience vectors created using the subjects are any good.
# 
# Again, lets save the results

# In[ ]:


with open("audience_subject_map.json", "w") as f:
    json.dump(audience_subject_map, f)
    
np.save("audience_embeddings_using_sub.npy", audience_embeddings_using_sub)


# ## Word2Vec Skip-Gram Audience Embedding
# 
# Another method we can try, is to use the skip-gram word2vec model to learn the vectors.

# In[ ]:


import gensim


# In[ ]:


model_audience = gensim.models.Word2Vec(
    size=EMBEDDING_SIZE, #10,
    window=99999,
    sg=1,
    hs=0,
    min_count=MIN_OCCURRENCES,
    workers=4,
    compute_loss=True
)


# In[ ]:


model_audience.build_vocab(news_data['audiences_tuples'].values)


# In[ ]:


get_ipython().run_cell_magic('time', '', "model_audience.train(\n    sentences=news_data['audiences_tuples'],\n    epochs=1,\n    total_examples=news_data.shape[0],\n    compute_loss=True,   \n)\n\nmodel_audience.get_latest_training_loss()")


# In[ ]:


model_audience.wv.similar_by_word("OIL")


# In[ ]:


model_audience.wv.similar_by_word("MTL")


# In[ ]:


model_audience.wv.similar_by_word("NZP")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'audience_word2vec_tsne = sklearn.manifold.TSNE(n_components=2).fit_transform(\n    model_audience.wv.vectors_norm\n)\nprint(audience_word2vec_tsne.shape)')


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.scatter(audience_word2vec_tsne[:, 0], audience_word2vec_tsne[:, 1], s=1)

for i, txt in enumerate(model_audience.wv.index2word):
    if i % 2 == 0:
        ax.annotate(txt, (audience_word2vec_tsne[i, 0], audience_word2vec_tsne[i, 1]))

plt.show()


# As before, it is no clearer if these results are any better.
# 
# Again, we will save the results.

# In[ ]:


with open("audience_skipgram.json", "w") as f:
    json.dump(model_audience.wv.index2word, f)
np.save("audience_skipgram_embeddings.npy", model_audience.wv.vectors_norm)


# ## Word2Vec Skip-Gram Subject Embedding
# 
# For completeness, lets learn vectors for the subjects using the skip-gram model.
# 
# WARNING: this will take several minutes (~15mins)

# In[ ]:


model_subject = gensim.models.Word2Vec(
    size=EMBEDDING_SIZE, #10,
    window=99999,
    sg=1,
    hs=0,
    min_count=MIN_OCCURRENCES,
    workers=4,
    compute_loss=True
)


# In[ ]:


model_subject.build_vocab(news_data['subjects_tuples'].values)


# In[ ]:


get_ipython().run_cell_magic('time', '', "model_subject.train(\n    sentences=news_data['subjects_tuples'],\n    epochs=1,\n    total_examples=news_data.shape[0],\n    compute_loss=True,   \n)\n\nmodel_subject.get_latest_training_loss()")


# In[ ]:


model_subject.wv.similar_by_word("FUND")


# In[ ]:


model_subject.wv.similar_by_word("EPMICS")
# COMDIS    Communicable Diseases
# SL        Sierra Leone


# In[ ]:


get_ipython().run_cell_magic('time', '', 'subjects_word2vec_tsne = sklearn.manifold.TSNE(n_components=2).fit_transform(model_subject.wv.vectors_norm)\nprint(subjects_word2vec_tsne.shape)')


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.scatter(subjects_word2vec_tsne[:, 0], subjects_word2vec_tsne[:, 1], s=1)

for i, txt in enumerate(model_subject.wv.index2word):
    if i % 20 == 0:
        ax.annotate(txt, (subjects_word2vec_tsne[i, 0], subjects_word2vec_tsne[i, 1]))

plt.show()


# Again, we will save the results

# In[ ]:


with open("subjects_skipgram.json", "w") as f:
    json.dump(model_subject.wv.index2word, f)
np.save("subject_skipgram_embeddings.npy", model_subject.wv.vectors_norm)

