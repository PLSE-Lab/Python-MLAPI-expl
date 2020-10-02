#!/usr/bin/env python
# coding: utf-8

# # Comparing Regions by Loan Use Descriptions
# 
# ## Contents
# 1. [Introduction](#introduction)
# 2. [Inspecting Data](#inspecting)
# 3. [Data Aggregation](#aggregate)
# 4. [Doc2Vec Mapping and Dimensionality Reduction](#doc2vec)
# 5. [Scatterplot Visualisation of Clusters](#vis)
# 6. [Case Study - Mali](#mali)
# 7. [Case Study - Guatemala](#guatemala)
# 8. [Around the Edge](#edge)
# 9. [Border-free Clustering](#borderless)
# 
# <a id='introduction'></a>
# ### Introduction
# 
# One of the goals of this data science challenge is to find out how the borrower situations differ from region to region, and capture such differences on a sufficiently granular level. When it comes to borrower situations, many factors must be considered, such as what they borrowers generally want in a region, what type of demand they are, how much they are asking for and how they reflect the local socio-economical situation. In this notebook, we will focus on figuring out what borrowers from different regions differ in terms of the intended use for their loans.

# ### Loading Packages and Data

# In[161]:


# load all libraries used in later analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (10, 6)
from matplotlib import style
style.use('ggplot')
import spacy
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, MeanShift, MiniBatchKMeans
import os
from pprint import pprint
import string
import re
from sklearn.decomposition import PCA
from collections import Counter


# In[2]:


# load data
loans = pd.read_csv("../input/kiva_loans.csv")


# <a id='inspecting'></a>
# ### Inspecting Data

# In[3]:


loans.head().transpose()


# We first look at the existing data fields in the *kiva_loans.csv* file. Remember that we wish to compare the difference in loan use cases between regions. We have three fields, "acitivity", "sector" and "use", that describe or imply the intended use for the loans. "Activity" and "sector" can be used to roughly profile each region for analysing the most common needs of loaners in each region, however it is the "use" field that contains the most information. Often, the "sector" and "actitvity" can be implied from the text in the "use" field. There are more details in the textual description, and they can be used to pick out the subtle differences between loans of similar actitity type and sectors. However, since this field is in raw text format, it is difficult to directly compare two loans or loans from two regions and determine how similar / different they are. There exist several NLP techniques such as key word frequency, term frequency - inverse document frequency (TF-IDF), topic modelling (LSA or LDA) or dense vector embeddings that can help us achieve such a comparison. Here, we will use the SpaCy NLP package to map regional use case descriptions into a dense vector space and attempt to identify regional differences in use cases through visualisation.

# <a id='aggregate'></a>
# ### Data Aggregation
# Since we are interested in comparing use cases between regions, we first aggregate the data by country and region, and then combine all use case descriptions in a region into a single text string. We are interested in how frequent certain words and phrases that characterise a particular use case appear within one region, therefore combining different loan records in one region will not lose the information we need. Later we may wish to further segregate the data by additional factors such as loaner gender, or weigh the use cases by loan amounts.

# In[4]:


# aggregate "use" by country and region and combine use text
use_by_CR = loans[['country', 'region', 'use']]     .replace(np.nan, "")     .groupby(['country', 'region'])['use']     .apply(lambda x: "\n".join(x))     .reset_index()  # normalise
use_by_CR['region'].replace("", "#other#", inplace=True)
use_by_CR['country'].replace("", "#other#", inplace=True)


# In[5]:


# generate a combined field for aggregation purposes
use_by_CR['CR'] = use_by_CR['country'] + "_" + use_by_CR['region']


#  <a id='doc2vec'></a>
#  ### Doc2Vec Mapping and Dimensionality Reduction
# Here we use SpaCy to calculate the document vectors for each concatenated use case descriptions. The basic intuition is that we use vectors in a high dimensional space to represent certain key words, and different combinations of these key words may characterise a certain use case, which will be represented by the combined direction of the individual word vectors. The document vector can be seen as a weighted average of the word and sentence vectors, and it can be compared to each key word or use case directions to evaluate how relevant a certain use case is to the region, or it can be used to compare the similarities of loaner intention characteristics between regions. 
# 
# ![](http://)We start by loading the SpaCy model and converting the texts.

# In[6]:


# now we use spacy to process the per-region use descriptions and obtain document vectors
nlp = spacy.load('en_core_web_lg', disable=["tagger", "parser", "ner"])

raw_use_texts = list(use_by_CR['use'].values)
processed_use_texts = [nlp(text) for text in raw_use_texts]


# In[7]:


processed_use_vectors = np.array([text.vector for text in processed_use_texts])
processed_use_vectors.shape


# As we see above, we have obtained the document vectors for use cases in each region. There are 300-dimensional vectors. While they might be quite useful for specific algorithms (especially neural network-based NLP algorithms), they are difficult to use for visualisation and pairwise comparisons. However, we may reduce the dimensionality of the data using common "flattening" algorithms such as TSNE or MDS. Here we will be using TSNE to reduce the dimensionality to 2.

# In[8]:


# we would like to map the document vectors into a space where we can compare their
# similarities and visualise them. A 2-D space is preferable. therefore we perform a TSNE
# transformation to flatten the 300-D vector space.
tsne = TSNE(n_components=2, metric='cosine', random_state=7777)
fitted = tsne.fit(processed_use_vectors)


# In[9]:


fitted_components = fitted.embedding_
fitted_components.shape


# In[10]:


use_by_CR['cx'] = fitted_components[:, 0]
use_by_CR['cy'] = fitted_components[:, 1]
use_by_CR.head()


# <a id='vis'></a>
# ### Scatterplot Visualisation
# Now that we have obtained a 2-D vector representation of the use cases in each region, we are able to plot them and visually examine regional similarities. We select the countries with most regions (n = 19), plot each individual regions and colour them by country.
# <a id='all'></a>

# In[11]:


# now we plot all the transformed points for each country against each other, for countries
# with the most recorded regions
country_region_cnt = use_by_CR.groupby('country').size()
selected_countries = country_region_cnt[country_region_cnt > 150]
n_selected_countries = len(selected_countries)
selected_country_pos = np.where(country_region_cnt > 150)[0]
id2country = dict(enumerate(selected_countries.index))
country2id = {v: k for k, v in id2country.items()}
selected_use_by_CR = use_by_CR.query('country in @selected_countries.index')


# In[12]:


fig, ax = plt.subplots(figsize=(16, 12))
plt.scatter(selected_use_by_CR['cx'], selected_use_by_CR['cy'], s=15,
            c=[country2id[x] for x in selected_use_by_CR['country']],
            cmap=plt.cm.get_cmap('tab20', 19))
formatter = plt.FuncFormatter(lambda val, loc: id2country[val])
plt.colorbar(ticks=np.arange(19), format=formatter);
plt.show()


# Here we see an interesting pattern. Often, regional use cases from one country are grouped together in a big cluster. However, some countries have a very wide spread (e.g. Phillippines), while others are broken down into multiple smaller clusters (e.g. Mali, Colombia). The clusters do not necessarily mean significant differences in use cases; they could simply indicate the differences between the description language used in the data field (e.g. a repeatedly used sentence format. Therefore, further investigation is needed to determine the implications of these clusters.

# <a id='mali'></a>
# ### Case Study 1 - Mali
# Let us look at Mali as an example. We can see on the graph that the use description for regions in Mali are roughly broken down into 3-4 clusters. Do these clusters indicate significant differences between the needs of loaners in different regions, or do they simply reflect the difference between the description language? We shall find out.
# 
# We first select all regions within Mali:

# In[13]:


# select all aggregated use cases in Mali:
mali_regional_uses = use_by_CR.query('country == "Mali"')
mali_regional_uses.shape


# We use a k-means algorithm to separate the regions of Mali into four groups. Here, k = 4 is chosen based on visual inspection. Higher k might result in finer details, but also more "artefact groups" that do not have much support.

# In[14]:


cluster = KMeans(n_clusters=4, random_state=7777)
cluster.fit_transform(mali_regional_uses[['cx', 'cy']]);


# We draw the clusters of Mali on the new scatterplot:

# In[15]:


fig, ax = plt.subplots(figsize=(16, 12))
plt.scatter(mali_regional_uses['cx'], mali_regional_uses['cy'], s=15,
            c=cluster.labels_,
            cmap=plt.cm.get_cmap('tab10', 4))
formatter = plt.FuncFormatter(lambda val, loc: "Cluser {0}".format(val))
plt.colorbar(ticks=np.arange(4), format=formatter);
plt.show()


# Now we print out some sample use case descriptions in each region to determine if the use cases are indeed similar within clusters and different between clusters.

# In[16]:


# let use see if the clusters are indeed different
# examples from cluster 0
for region_uses in mali_regional_uses['use'].iloc[cluster.labels_ == 0].iloc[:10]:
    print(region_uses[:min(500, len(region_uses))], end="")
    print('...' if len(region_uses) > 500 else "")
    print('-' * 20)


# The common theme of this cluster is "pay for labour (during farming)" and sometimes "buy farming supplies".

# In[17]:


# examples from cluster 1
for region_uses in mali_regional_uses['use'].iloc[cluster.labels_ == 1].iloc[:10]:
    print(region_uses[:min(500, len(region_uses))], end="")
    print('...' if len(region_uses) > 500 else "")
    print('-' * 20)


# Cluster 1 is quite different from cluster 0 in that it mostly contains use cases regarding buying food, clothing and other necessities, either for self-use or resale.

# In[18]:


# examples from cluster 2
for region_uses in mali_regional_uses['use'].iloc[cluster.labels_ == 2].iloc[:10]:
    print(region_uses[:min(500, len(region_uses))], end="")
    print('...' if len(region_uses) > 500 else "")
    print('-' * 20)


# Cluster 2 is mostly about buying fertiliser, seeds and other farming supplies. Notice that there are some content overlap with cluster 0. Although Cluster 0 and 2 appear to be quite separated on the graph above, they are still relatively close in the all countres graph and on the same side blob as well.

# In[19]:


# examples from cluster 3
for region_uses in mali_regional_uses['use'].iloc[cluster.labels_ == 3].iloc[:10]:
    print(region_uses[:min(500, len(region_uses))], end="")
    print('...' if len(region_uses) > 500 else "")
    print('-' * 20)


# Cluster 3 is not as "pure" as the other clusters, as it is mostly a "leftover" cluster in k-means, yet we can still see some common themes such as "buying animals and seeds for resell".
# 
# [*back to top countries scatterplot*](#all)
# 
# As we see above, the clusters we identified indeed capture the loan intention differences between regions within a country. What about between countries? Can it identify similar needs between groups of regions in two distinct countries?

# <a id='guatemala'></a>
# ### Case Study 2 - Guatemala
# We noticed in the top countries plot that one of the clusters of Guatemala is quite close to the right side cluster of Mali. Are they indeed similar in intended loan use cases? We shall find out. First we repeat the analysis in the last section:

# In[20]:


guatemala_regional_uses = use_by_CR.query('country == "Guatemala"')
guatemala_regional_uses.shape


# In[21]:


cluster2 = KMeans(n_clusters=5, random_state=7777)
cluster2.fit_transform(guatemala_regional_uses[['cx', 'cy']]);


# In[22]:


fig, ax = plt.subplots(figsize=(16, 12))
plt.scatter(guatemala_regional_uses['cx'], guatemala_regional_uses['cy'], s=15,
            c=cluster2.labels_,
            cmap=plt.cm.get_cmap('tab10', 5))
formatter = plt.FuncFormatter(lambda val, loc: "Cluser {0}".format(val))
plt.colorbar(ticks=np.arange(5), format=formatter);
plt.show()


# In[23]:


for region_uses in guatemala_regional_uses['use'].iloc[cluster2.labels_ == 1].iloc[:10]:
    cleaned = re.sub(r'(\n\s*)+\n+', '\n', region_uses)  # remove excessive empty lines 
    # caused by missing data
    print(cleaned[:min(500, len(cleaned))], end="")
    print('...' if len(cleaned) > 500 else "")
    print('-' * 20)


# [*back to top countries scatterplot*](#all)
# 
# [*back to Mali*](#mali)
# 
# Here, we see that it is similar to cluster 1 of Mali in that they are both mostly about buying necessities and other everyday products, but it has a bigger emphasis one resell. Let us put this cluster and the cluster 1 of Mali together:

# In[24]:


combined_data = pd.concat([guatemala_regional_uses.iloc[cluster2.labels_ == 1],
                          mali_regional_uses.iloc[cluster.labels_ == 1]], axis=0)
i2c = {0: "Mali", 1: "Guatemala"}
c2i = {v: k for k, v in i2c.items()}


# In[25]:


fig, ax = plt.subplots(figsize=(16, 12))
plt.scatter(combined_data['cx'], combined_data['cy'], s=15,
            c=[c2i[x] for x in combined_data['country']],
            cmap=plt.cm.get_cmap('tab10', 2))
formatter = plt.FuncFormatter(lambda val, loc: i2c[val])
plt.colorbar(ticks=np.arange(2), format=formatter);
plt.show()


# As we see, the main mass of these two clusters are very close together but not overlapping. Maybe the degree of emphasis on "resale" is the characteristic that keep them apart. Let us do a simple word count:

# In[26]:


def count_resales(s):
    resales_words = ["resell", "sell", "resale"]
    return len(re.findall("|".join(["(?:{0})".format(x) for x in resales_words]), s))


# In[27]:


print("Mentions of resales in Mali:")
print(mali_regional_uses.iloc[cluster.labels_ == 1]["use"].apply(count_resales).sum())


# In[28]:


print("Mentions of resales in Guatemala:")
print(guatemala_regional_uses.iloc[cluster2.labels_ == 1]["use"].apply(count_resales).sum())


# As we see, indeed, there are much more mentions of "resale" and the likes in the Guatemala cluster. However, at this point we are not able to determine whether this difference caused the separation of clusters among all other possible causes. We will have to go back to higher dimensions and find out the direction that is responsible for the words like "sell", "resell" or "resale".

# <a id='edge'></a>
# ### Around the Edge
# We noticed that many of the countries have dense concentuations of points at the edges of the graph. Does this mean that people from these countries have unique uses for their loans that are not common elsewhere? Can we also use this to identify parts of a country with unusual needs?
# 
# [*back to top countries scatterplot*](#all)
# 
# Let us look at **Samoa** at first. The point masses for the country are almost entirely concentuated on the bottom edge:

# In[29]:


samoa = use_by_CR.query('country == "Samoa"')
fig, ax = plt.subplots(figsize=(16, 12))
plt.scatter(samoa['cx'], samoa['cy'], s=15)
plt.show()


# In[30]:


for region_uses in samoa['use'][:10]:
    cleaned = re.sub(r'(\n\s*)+\n+', '\n', region_uses)
    print(cleaned[:min(500, len(cleaned))], end="")
    print('...' if len(cleaned) > 500 else "")
    print('-' * 20)


# From a few samples of use cases, we see that what Samoans are looking for are not particularly exotic, but the use case descriptions are oddly very specific compared to what we see earlier, typically with a long list of desired items. This is not exactly what we expected. It does remind us that not everything captured by the document vectors relate to semantic differences in our area of interest. 
# 
# [*back to top countries scatterplot*](#all)
# 
# Now let us see if there is something unique about use cases from **Kyrgyzstan** (also on the edge of the top):

# In[31]:


kyrg = use_by_CR.query('country == "Kyrgyzstan"')
for region_uses in kyrg['use'][:10]:
    cleaned = re.sub(r'(\n\s*)+\n+', '\n', region_uses)
    print(cleaned[:min(500, len(cleaned))], end="")
    print('...' if len(cleaned) > 500 else "")
    print('-' * 20)


# As we see, the use cases are mostly livestock-focused (especially sheep and cows) with a smaller portion for farming. Let us check and see if Kyrgyzstan has the highest percentage of livestock-related use cases:

# In[32]:


def count_livestock(s):
    resales_words = ["sheep", "cow", "calf", "calves", "bull", "livestock"]
    return len(re.findall("|".join(["(?:{0})".format(x) for x in resales_words]), s))


# In[33]:


livestock_counts = use_by_CR.copy()
livestock_counts['n_words'] = livestock_counts['use']     .apply(lambda x: len(re.findall(r"\w+", x)))
livestock_counts['n_livestock'] = livestock_counts["use"].apply(count_livestock)


# In[34]:


livestock_totals = livestock_counts.groupby('country')[['n_words', 'n_livestock']].sum()
livestock_totals['ratio'] = livestock_totals['n_livestock'] / livestock_totals['n_words']


# In[35]:


livestock_totals.sort_values("ratio", ascending=False)[:10]


# Indeed, Kyrgyzstan easily leads in terms of livestock-related words to total words ratio. It is likely that this is the distinguishing factor that put Kyrgyzstan on the edge of the graph, away from the other countries and regions.

# <a id="borderless"></a>
# ### Border-free Clustering (WIP)
# Now that we know that the use case descriptions can be used to characterise borrower needs within a region, we may exploit this new information to break down the country borders and bring regions of similar needs together (figuratively).  We will apply a clustering algorithm at region level in a higher dimensional document vector space.

# In[38]:


print("document vectors dimensions: {0}".format(processed_use_vectors.shape))


# We have 300-D document vectors for the regional use case descriptions. It is still pretty high dimensional for clustering tasks. Therefore, we will employ PCA first to reduce it to a more manageable number.

# In[47]:


pca = PCA(n_components=100, random_state=7777)
pca.fit(processed_use_vectors)


# Explained variance ratio of principle components:

# In[48]:


N = 20
sns.barplot(x=pca.explained_variance_ratio_[:N], y=["C_{0}".format(x) for x in range(N)])


# We take the principle components that together explain 80% of the variance. This is not perfect, but should still allow us to obtain some meaningful clustering results.

# In[56]:


N = 35
print("number of selected PCs: {}".format(N))
print("total % variance explained: {}".format(np.sum(pca.explained_variance_ratio_[:N])))


# In[61]:


low_dim_vecs = pca.transform(processed_use_vectors)[:, :N]
print(low_dim_vecs.shape)


# In[168]:


ldcls = MiniBatchKMeans(n_clusters=50, random_state=7777)
ldcls.fit(low_dim_vecs);
use_by_CR['cluster'] = ldcls.labels_


# In[169]:


print(Counter(ldcls.labels_))


# In[170]:


N = len(np.unique(ldcls.labels_))
fig, ax = plt.subplots(figsize=(16, 12))
plt.scatter(use_by_CR['cx'], use_by_CR['cy'], s=15,
            c=ldcls.labels_ + 1,
            cmap=plt.cm.get_cmap('rainbow', N))
formatter = plt.FuncFormatter(lambda val, loc: "C_{}".format(val))
plt.colorbar(ticks=np.arange(N), format=formatter);
plt.show()


# ### Plot the Clusters on the World Map

# In[ ]:




