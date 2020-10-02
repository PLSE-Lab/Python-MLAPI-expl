#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import nltk
from nltk.corpus import stopwords
import seaborn as sns
import os
os.chdir('../input')


# In[ ]:


with open('china.xml') as f:
    f = f.read()
    these_regex="<title.*?>(.+?)</title>"
    pattern=re.compile(these_regex)
    china_titles=re.findall(pattern,f)
    
china_words = []
    
for t in china_titles:
    t = t.split(" ")
    for w in t:
        if w not in stopwords.words('english'):
            china_words.append(w)


# In[ ]:


print(china_words)


# In[ ]:


china_word_dist = nltk.FreqDist(w.lower() for w in china_words)


# In[ ]:


china_word_dist.most_common(20)


# In[ ]:


with open('europe.xml') as f:
    f = f.read()
    these_regex="<title.*?>(.+?)</title>"
    pattern=re.compile(these_regex)
    europe_titles=re.findall(pattern,f)

europe_words = []
    
for t in europe_titles:
    t = t.split(" ")
    for w in t:
        if w not in stopwords.words('english'):
            europe_words.append(w)


# In[ ]:


print(europe_words)


# In[ ]:


europe_word_dist = nltk.FreqDist(w.lower() for w in europe_words)


# In[ ]:


europe_word_dist.most_common(20)


# In[ ]:


europe_words_most_common = set([s[0] for s in europe_word_dist.most_common(20)])


# In[ ]:


china_words_most_common = set([s[0] for s in china_word_dist.most_common(20)])


# In[ ]:


europe_common_exclusive_words = europe_words_most_common- china_words_most_common


# In[ ]:


china_common_exclusive_words = china_words_most_common - europe_words_most_common


# In[ ]:


features = list(europe_common_exclusive_words) + list(china_common_exclusive_words)


# In[ ]:


len(features)


# In[ ]:


features


# In[ ]:


china_vectors = []
for title in china_titles:
    current_vector = [0,0]
    for word in title.lower().split(" "):
        if word in china_common_exclusive_words:
            current_vector[0] += 1
        if word in europe_common_exclusive_words:
            current_vector[1] += 1
    china_vectors.append(current_vector)


# In[ ]:


china_vectors


# In[ ]:


europe_common_exclusive_words


# In[ ]:


europe_vectors = []
for title in europe_titles:
    current_vector = [0,0]
    for word in title.lower().split(" "):
        if word in china_common_exclusive_words:
            current_vector[0] += 1
        if word in europe_common_exclusive_words:
            current_vector[1] += 1
    europe_vectors.append(current_vector)


# In[ ]:


europe_vectors


# In[ ]:


relevant_china_vectors = [v for v in china_vectors if sum(v) > 0]
relevant_europe_vectors = [v for v in europe_vectors if sum(v) > 0]
all_vectors = china_vectors + europe_vectors
all_relevant_vectors = relevant_china_vectors + relevant_europe_vectors


# In[ ]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
kmeans.fit(all_vectors)


# In[ ]:


kmeans = KMeans(n_clusters=2)

kmeans.fit(all_relevant_vectors)


# In[ ]:


kmeans.predict(all_relevant_vectors)


# In[ ]:


all_relevant_vectors


# In[ ]:


results = kmeans.predict(all_relevant_vectors)
gg = list(zip(all_relevant_vectors, results))


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.scatterplot(x=[x[0][0] for x in gg],
                y=[x[0][1] for x in gg],
                hue=[x[1] for x in gg],
                size=[gg.count(x) for x in gg],
                s=10).set_title("Word Bucket Distribution")


# In[ ]:


import random
europe_vectors = []
for title in europe_titles:
    current_vector = [0,0,title]
    for word in title.lower().split(" "):
        if word in china_common_exclusive_words:
            current_vector[0] += random.random()
        if word in europe_common_exclusive_words:
            current_vector[1] += random.random()
    europe_vectors.append(current_vector)
    
china_vectors = []
for title in china_titles:
    current_vector = [0,0,title]
    for word in title.lower().split(" "):
        if word in china_common_exclusive_words:
            current_vector[0] += random.random()
        if word in europe_common_exclusive_words:
            current_vector[1] += random.random()
    china_vectors.append(current_vector)


# In[ ]:


import random
random.seed(999)

europe_vectors = []
for title in europe_titles:
    current_vector = [0,0]
    for word in title.lower().split(" "):
        if word in china_common_exclusive_words:
            current_vector[0] += random.random()
        if word in europe_common_exclusive_words:
            current_vector[1] += random.random()
    europe_vectors.append(current_vector)
    
china_vectors = []
for title in china_titles:
    current_vector = [0,0]
    for word in title.lower().split(" "):
        if word in china_common_exclusive_words:
            current_vector[0] += random.random()
        if word in europe_common_exclusive_words:
            current_vector[1] += random.random()
    china_vectors.append(current_vector)


# In[ ]:


relevant_china_vectors = [v for v in china_vectors if sum(v[:2]) > 0]
relevant_europe_vectors = [v for v in europe_vectors if sum(v[:2]) > 0]
all_vectors = china_vectors + europe_vectors
all_relevant_vectors = relevant_china_vectors + relevant_europe_vectors


# In[ ]:


all_relevant_vectors


# In[ ]:


kmeans = KMeans(n_clusters=2)

kmeans.fit(all_relevant_vectors)

kmeans.predict(all_relevant_vectors)


# In[ ]:


results = kmeans.predict(all_relevant_vectors)
gg = list(zip(all_relevant_vectors, results))


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.scatterplot(x=[x[0][0] for x in gg],
                y=[x[0][1] for x in gg],
                hue=[x[1] for x in gg],
                size=[gg.count(x) for x in gg],
                s=10).set_title("Word Bucket Distribution")


# In[ ]:


# https://plot.ly/~marcusbao12/1/

