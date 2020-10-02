#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd

sessions = pd.read_csv("/kaggle/input/money-2020-sessions/sessions.csv")
sessions = sessions[sessions["title"].apply(lambda x: x is not None and isinstance(x, str))]
sessions = sessions[sessions["content"].apply(lambda x: x is not None and isinstance(x, str))]


# In[ ]:


sessions.head(5)


# ## Tag Statistics

# In[ ]:


def make_tag_list(data):
    if "session_tags" in data.columns:
        _tags = data["session_tags"]
    else:
        _tags = data["tags"].apply(lambda x: [] if not isinstance(x, str) else x.split(","))

    tags = []
    session_tags = []
    for i, t in _tags.iteritems():
        s_tags = []
        for n in t:
            n = n.strip()
            if n.startswith("Level") or n.startswith("Venetian") or n.endswith("Stage") or n.endswith("Lounge")                 or n.lower().endswith("room") or n.lower().endswith(" hall") or n.endswith("Garden"):
                # Prece tag
                n = ""
            elif n.endswith("Presentation") or n.endswith("Keynote"):
                n = "_Presentation"
            elif n.startswith("Workshop") or n == "Case Studies":
                n = "_Study"
            elif n.endswith("Discussion") or n.startswith("Debate") or                     n.startswith("Networking") or n == "Fireside Chat":
                n = "_Communication"

            if n:
                tags.append(n)
                if not n.startswith("_"):
                    s_tags.append(n)
        
        session_tags.append(s_tags)
    
    if "session_tags" not in data.columns:
        data["session_tags"] = session_tags
    
    return pd.Series(tags)

make_tag_list(sessions).value_counts().sort_values(ascending=True).plot.barh(figsize=(5,10))


# ## Analysis of title and session content

# In[ ]:


get_ipython().system('python -m spacy download en_core_web_lg')


# In[ ]:


import spacy


class NounTokenizer(object):

    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")

    def __call__(self, doc):
        nouns = []
        propn = []

        if doc is None or not isinstance(doc, str):
            return nouns

        tokens = self.nlp(doc)
        for t in tokens:
            if t.pos_ == "NOUN":
                nouns.append(t.lemma_.lower())
            elif t.pos_ == "PROPN":
                propn.append(t.text.lower())
            else:
                if len(propn) > 0:
                    nouns.append(" ".join(propn))
                    propn.clear()

        return nouns


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


def make_noun_list(data, keyword_filter=()):
    tokenizer = NounTokenizer()
    vectorizer = CountVectorizer(
                    stop_words="english", max_df=0.7, ngram_range=(1, 2),
                    tokenizer=tokenizer)
    
    targets = data[data["session_tags"].apply(lambda x: len(x) > 0)]
    docs = pd.concat((targets["title"], targets["content"]), ignore_index=True, axis=0)
    vectorizer.fit(docs)
    words = vectorizer.get_feature_names()

    if len(keyword_filter) == 0:
        counts = vectorizer.transform(docs).sum(axis=0).tolist()[0]
    else:
        vectors = vectorizer.transform(docs)
        keyword_count = np.zeros(vectors.shape[0])

        for k in keyword_filter:
            if k in words:
                keyword_count += vectors[:, words.index(k)].toarray().flatten()

        vectors = vectors[keyword_count > 0, :]
        counts = vectors.sum(axis=0).tolist()[0]

    vocab = {}
    for i, w in enumerate(words):
        vocab[w] = counts[i]

    return pd.Series(vocab)

make_noun_list(sessions).nlargest(30).sort_values(ascending=True).plot.barh(figsize=(5,10))


# In[ ]:


make_noun_list(sessions, keyword_filter=("ai", "machine", "fintech")).nlargest(30).sort_values(ascending=True).plot.barh(figsize=(5,10))


# In[ ]:


make_noun_list(sessions, keyword_filter=("esg")).nlargest(30).sort_values(ascending=True).plot.barh(figsize=(5,10))


# In[ ]:


class KeyWordFilter():
    
    def __init__(self, keywords, excludes=(), filter_by_session=True):
        self.tokenizer = NounTokenizer()
        self.keywords = keywords
        self.excludes = excludes
        self.filter_by_session = filter_by_session

    def __call__(self, x):
        tokens = []
        for key in ("title", "content"):
            tokens += self.tokenizer(x[key])
        
        if self.filter_by_session and len(x["session_tags"]) == 0:
            return False
        
        excludes = False
        for k in self.excludes:
            if k in tokens:
                excludes = True
                break
        
        if excludes:
            return excludes

        is_include = False
        for k in self.keywords:
            if k in tokens:
                is_include = True
                break
        
        return is_include


# In[ ]:


kfilter = KeyWordFilter(
    keywords=["ai", "machine", "fintech"],
    excludes=["blockchain"]
)
ai_relateds = sessions[sessions.apply(kfilter, axis=1)][["title", "content", "session_tags"]]
print(len(ai_relateds))


# In[ ]:


ai_relateds


# In[ ]:


ai_relateds["title"].tolist()


# In[ ]:


make_tag_list(ai_relateds).value_counts().sort_values(ascending=True).plot.barh(figsize=(5,10))


# In[ ]:


kfilter_e = KeyWordFilter(
    keywords=["esg"]
)
esg_relateds = sessions[sessions.apply(kfilter_e, axis=1)][["title", "content", "session_tags"]]
print(len(esg_relateds))


# In[ ]:


ai_relateds[ai_relateds["session_tags"].apply(lambda x: "Regulation & Regtech" in x)]


# In[ ]:


ai_relateds[ai_relateds["session_tags"].apply(lambda x: "Banking & PFM" in x)]

