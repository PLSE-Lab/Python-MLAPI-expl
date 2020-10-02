#!/usr/bin/env python
# coding: utf-8

# **1.Importing the basic libraries and walking through the source directory containing the files.**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **2. Reading the 'A million headline' dataset**

# In[ ]:


data = pd.read_csv('/kaggle/input/million-headlines/abcnews-date-text.csv')
data.head()


# **3. Creating the Publish Year, Publish Month and Publish Day. **

# In[ ]:


data['publish_year'] = data['publish_date'].apply(lambda x:int(x/10000))
data['publish_month'] = data['publish_date'].apply(lambda x:int(((x)%10000)/100))
data['publish_day'] = data['publish_date'].apply(lambda x:((x)%10000)%100)


# **4. Plotting Number of Headlines every year**

# In[ ]:


import matplotlib.pyplot as plt
plt.hist(data['publish_year'], facecolor='blue', alpha=0.8, rwidth = 0.5)
plt.xlabel('Year')
plt.ylabel('#News Headlines')
plt.title('#News Headlines in each year')
plt.show()


# **5. Plotting Number of Headlines published every month**

# In[ ]:


plt.hist(data['publish_month'],12, facecolor='blue', alpha=0.8, rwidth = 0.5)
plt.xlabel('Month')
plt.ylabel('#News Headlines')
plt.title('#News Headlines in each month')
plt.show()


# **6. Plotting Number of Headlines published by every day of month**

# In[ ]:


plt.hist(data['publish_day'],31, facecolor='blue', alpha=0.8, rwidth = 0.5)
plt.xlabel('Day')
plt.ylabel('#News Headlines')
plt.title('#News Headlines on each day of month')
plt.show()


# **7. Creating text corpus for word cloud and frequency plots**

# In[ ]:


corp = str()
for i in range(len(data['headline_text'])):
    corp += (' ')+data['headline_text'][i]


# **8. Tokenize the corpus**

# In[ ]:


import nltk
words = nltk.word_tokenize(corp)
#data['headline_text'][1] + (' ') + data['headline_text'][2] + data['headline_text'][3]


# **9. Cleaning the corpus - removing stopwords & punctuations**

# In[ ]:


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
f_words = [w for w in words if not w in stop_words] 

punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
fp_words = [w for w in f_words if not w in punctuations] 


# **10. Creating the Frequency dictionary and frequency plots**

# In[ ]:


fd = nltk.FreqDist(fp_words)

df_fdist = pd.DataFrame.from_dict(fd, orient='index')
df_fdist.columns = ['Frequency']
df_fdist.index.name = 'Term'

freq_df = df_fdist[df_fdist['Frequency']>500]
d = freq_df.to_dict()['Frequency']

#plt.figure(figsize=(20, 8))
freq_df1 = df_fdist[df_fdist['Frequency']>7500]
freq_df1.sort_values('Frequency',ascending=False).plot(kind='bar')
#freq_df1.columns


# **11. Creating Word Cloud**

# In[ ]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.figure(figsize=(32,32))
wordcloud = WordCloud()
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# **12. Cleaner function to apply on the 'headline_text' column for modeling preparation**

# In[ ]:


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import nltk
def text_cleaner(text):
    stop_words = set(stopwords.words('english'))
    f_words = [w for w in nltk.word_tokenize(text) if not w in stop_words] 
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    fp_words = [w for w in f_words if not w in punctuations] 
    fp_words_stem = [ps.stem(words) for words in fp_words]
    fp_sent = ' '.join(word for word in fp_words_stem)
    return fp_sent

#text_cleaner(data['headline_text'][1])


# **13. Applying cleaning function to 'headline_text'**

# In[ ]:


data['headline_text_clean'] = data['headline_text'].apply(text_cleaner)
data['headline_text_clean'][:10]


# **14. Implementing TF-IDF on the data with capping the number of features / token to 10000. The dimension of TF-IDF is 11,03,663 X 10000**

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=10000)
vectorizer.fit(data['headline_text_clean'].values)
data_tfidf = vectorizer.transform(data['headline_text_clean'])

tfidf_to_word = np.array(vectorizer.get_feature_names())
#tfidf_to_word


# **15. Decomposing the TF-IDF matrix to smaller matrices W (11,03,663 X 50) and H (50 X 10000) using Non-Negative Matrix Factorization**

# In[ ]:


from sklearn.decomposition import NMF
nmf = NMF(n_components=50, solver="mu")
W = nmf.fit_transform(data_tfidf)
H = nmf.components_
#W.shape
#H.shape


# **16. Each topic is a made of collection of words. Matrix H represents each topic (50 rows/topics) and words making it (10000 columns/words). Here we are printing the top 10 words making the topic.**

# In[ ]:


for i, topic in enumerate(H):
    print("Topic {}: {}".format(i + 1, ",".join([str(x) for x in tfidf_to_word[topic.argsort()[-10:]]])))


# **17. Each document can be represented as a collection of topics. Matrix W represents each document (10000 rows/documents subsetted here) and topics making it (50 columns/topics). Here we are printing the top 10 topics making the document.**

# In[ ]:


topic_list = np.array([("Topic" + " "+ str(i)) for i in range(1,51)])
for i,topic in enumerate(W[:10000,]):
    print("Headline {}: {}".format(i+1,",".join([str(x) for x in topic_list[topic.argsort()[-10:]]])))


# **18. Implementing t-SNE to reduce the W (50 dim vectors) to 2 dims for plotting**

# In[ ]:


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(W[:10000,])

df_subset = pd.DataFrame()
df_subset.head()

import matplotlib.pyplot as plt
import seaborn as sns
df_subset = pd.DataFrame()
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]


# **19. t-SNE option 2 for plotting **

# In[ ]:


plt.figure(figsize=(16,10))
plt.scatter(x=tsne_results[:,0],y=tsne_results[:,1],alpha=0.8, c="y")

plt.figure(figsize=(16,10))
x = x=tsne_results[:,0]
y = y=tsne_results[:,1]
plt.scatter(x,y,alpha=0.8, c=y)


# **20. Implement k-Means. 10 clusters is fast but 100 clusters gives better results. ** 

# In[ ]:


from sklearn.cluster import KMeans

# Number of clusters
kmeans = KMeans(n_clusters=10)
# Fitting the input data
kmeans = kmeans.fit(W)
# Getting the cluster labels
labels = kmeans.predict(W)
# Centroid values
centroids = kmeans.cluster_centers_


# **21. Getting results from clustering**

# In[ ]:


centroids
data['cluster'] = labels
data['cluster'].value_counts()
data[['headline_text_clean','cluster']].sample(n=1000)


# **22. Plotting Inertial vs Number of Clusters. If k belongs to np.arange(1,102,0), it runs k-Means with k=1,11,21,....,101. Larger k might take 15 mins+ to run. **

# In[ ]:


from sklearn.cluster import KMeans
import numpy as np
sse = {}
for k in np.arange(1, 10, 1):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(W)
    #data["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# **23. Finding the cluster membership of the input query for recommendation engine. ANd filtering all the headlines belonging to that clusters. The search for recommendations would happen over the members of only this cluster and not globally.**

# In[ ]:


clust = data['cluster'][2]
W_filter = data[data['cluster'] == clust].index
W_filter

W[W_filter].shape


# **24. A function to find cosine similarity for a given input query with all the other elements. The top 10 similar elements would be shown as recommendations for the input element. ** 

# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity

def cos_pair(X):
    sim_x = []
    for i in range(len(W[W_filter])):
        sim_x.append(cosine_similarity(X.reshape(1,W.shape[1]),W[W_filter][i].reshape(1,W.shape[1])))
    return sim_x
        
sim = cos_pair(W[0])
top10recos = sorted(range(len(sim)), key=lambda i: sim[i])[-10:]
data['headline_text'][top10recos]

