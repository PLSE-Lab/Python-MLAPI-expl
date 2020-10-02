#!/usr/bin/env python
# coding: utf-8

# # Unsupervised learning of TED talk categories
# 
# In this notebook I will primarily use sklearn to group TED talks based on metadata keywords. My goal is to present multiple ways of clustering data as an introduction to some of the options available in sklearn. There are far more advanced tools available for NLP, such as specific libraries like spaCy, NLTK, and gensim, but my goal here is sipmly to introduce some basic tools available right in sklearn for text classification and clustering.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import ast

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

from wordcloud import WordCloud
from wordcloud import get_single_color_func

from collections import Counter


SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE) 

cmap_string = 'Spectral'


# For this analysis I will only consider the talk metadata.

# In[ ]:


# data downloaded from Kaggle: https://www.kaggle.com/rounakbanik/ted-talks
df = pd.read_csv('../input/ted-talks/ted_main.csv')

df['date'] = pd.to_datetime(df.film_date, unit='s')

df.head(3)


# In[ ]:


print(df.shape)


# We see our dataframe has 2550 observations.  
# 
# First let's clean up the talk keyword tags.

# In[ ]:


df.tags = df.tags.apply(ast.literal_eval)
df['tags_single_string'] = df.tags.apply(' '.join)


# Next we will use CountVectorizer and TfidfTransformer applied to each talk's set of keywords. CountVectorizer essentially counts the number of times each word appears, and TfidfTransformer inversely weights each word by how frequently it occurs across all talks' metadata.

# In[ ]:


# get word counts and TF-IDF arrays from tags
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

X = df.tags_single_string.values

print(f'The first talk keywords: [{X[0]}]')

countv = CountVectorizer()
X_counts = countv.fit_transform(X)

tfidf = TfidfTransformer()
X_tfidf = tfidf.fit_transform(X_counts)


# In[ ]:


print(X_tfidf.shape)
print(countv.get_feature_names()[:10])


# We see our transformed keyword matrix (X_tfidf) has 2550 observations, matching our original dataframe, and 440 features, each of which correspond to a unique word. There are a total of 440 unique words across all the talk keyword sets.

# A common first step in unsupervised learning is to reduce the dimensions of your data. This will allow us to plot the data in 2 dimensions to visualize it and get a sense of what is going on. Principal component analysis (PCA) is a method which finds the axes of highest variance in the data and rotates the data so that the first dimension corresponds to the greatest variance, the second dimension corresponds to the second greatest variance, and so on.
# 
# Let's first run PCA on our data to visualize the distribution of talks in this dimensionally reduced PC space.

# In[ ]:


pca_comps = 2
pca = PCA(n_components=pca_comps)
x_pca = pca.fit_transform(X_tfidf.toarray())

fig, ax = plt.subplots(1,1, figsize=(5,5))
scatter, = ax.scatter(x_pca[:,0], x_pca[:,1])
ax.legend(*scatter.legend_elements())
ax.set_xlabel(f'PC 1 ({100*pca.explained_variance_ratio_[0]: 0.2f}%)')
ax.set_ylabel(f'PC 2 ({100*pca.explained_variance_ratio_[1]: 0.2f}%)')
plt.show()


# It looks like the data points form a central cluster with 3 vertices. 
# 
# But what do principal component 1 (PC 1) and principal component 2 (PC 2) actually represent? They represent the the new dimensions which were found by PCA to be a linear combination of the original dimensions, i.e., individual words! The loadings of each PC correspond to how much each word in our original vocabulary contribute to each principal component.
# 
# Let's look at the words with the highest and lowest loadings in PC 1 and PC 2 to understand what these first two principal components represent in the data.

# In[ ]:


feature_names = countv.get_feature_names()
words_to_grab = 10

fig, ax = plt.subplots(1, 2, figsize=(12, 10))
for i, comp in enumerate(pca.components_):
    top_words = [feature_names[ind] for ind in comp.argsort()[-1:-words_to_grab:-1]]
    bottom_words = [feature_names[ind] for ind in comp.argsort()[words_to_grab::-1]]
    bp_top = ax[i].barh(top_words, comp[comp.argsort()[-1:-words_to_grab:-1]], 
              color='blue', edgecolor='black')
    bp_bottom = ax[i].barh(bottom_words, comp[comp.argsort()[words_to_grab::-1]], 
              color='red', edgecolor='black')
    ax[i].axvline(0, linestyle='--', color='black')
    ax[i].set_title(f'Top word loadings for PC {i+1}')
    ax[i].set_xlabel(f'PC {i+1} loading')
    ax[i].invert_yaxis()
plt.subplots_adjust(wspace=0.5)
plt.show()


# As we can see from the words contributing the most positive and most negative loadings, PC 1 seems to represent an art vs. activism axis, with words like music, entertainment, and creativity contributing to a positive score. On the other hand, words like global, issues, economics, and change contribute to a negative score. 
# 
# PC 2 seems to represent something like a technical vs. non-technical axis, with words like science, medicine, and research contributing to positive scores and words like global, music, issues, and politics contributing to negative scores. 
# 
# Now we have a baseline idea of what sort of talks to expect in each region of PC space.
# 

# In[ ]:


# Here I define a few helper functions which will be useful throughout the rest of the notebook.

def assign_word_to_cluster(string_list, labels, vocab):
    '''generate a counter to get word counts per cluster, 
    then assign each word to the cluster it's most often associated with'''
    
    texts = [' '.join(string_list.loc[labels==c].values).split(' ') 
             for c in list(set(labels))]

    counters = [Counter(text) for text in texts]

    word_cluster = {}
    for word in vocab:
        cluster_count = []
        for i, c in enumerate(counters):
            cluster_count.append(c[word]/len(np.where(labels == i)))
            word_cluster[word] = np.argmax(cluster_count)

        cluster_lookup = {k: [] for k in range(1+np.max(labels))}
        for k, v in word_cluster.items():
            cluster_lookup[v].append(k)
    return word_cluster, cluster_lookup

def assign_word_to_color(cluster_lookup, cmap):
    ''' map each cluster's associated word to a color;
    return a dictionary of {hex color}: {word_list} for passing to word cloud API'''
    color_dict = {}
    for i, cluster in enumerate(list(cluster_lookup.keys())):
        color_dict[colors.to_hex(cmap(i))] = cluster_lookup[cluster]
    return color_dict

def plot_pc_space(x_pca, pca, labels, pca_comps, cmap):
    ''' plot observations in PC 1-2 space, colored by kmeans label'''
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
    scatter = ax.scatter(x_pca[:,0], x_pca[:,1], c=labels, cmap = cmap)
    #ax.legend(*scatter.legend_elements(), title='Clusters')
    ax.set_xlabel(f'PC 1 ({100*pca.explained_variance_ratio_[0]: 0.2f}%)')
    ax.set_ylabel(f'PC 2 ({100*pca.explained_variance_ratio_[1]: 0.2f}%)')
    ax.set_title(f'{pca_comps} PCs')
    plt.show()
    
# the following function is from the WordCloud API documentation for applying colors to specific words: https://amueller.github.io/word_cloud/index.html
class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)


# Next let's try to find the clusters in which talks naturally fall. From visualizing the data in PC space, we can see that that there don't seem to be discrete clusters, but we can still try to demarcate regions that contain talks of a similar broad category, like art, science, or politics. 
# 
# K-means clustering is an unsupervised learning algorithm that finds k clusters in a data set by minimizing the Euclidean distance of each point to its assigned cluster center.
# 
# So we can use K-means to cluster our dimensionally reduced data, but one question is how to set the number of clusters, k, in the K-means algorithm. One way is to find the average silhouette score across all observations and choose k such that the silhouette score is maximized. The silhouette score is described in the sklearn documentation, but it is essentially weighs the inter vs. intra-cluster distances. A high score indicates more unambiguous cluster assignments, and low scores indicates ambiguous or incorrect cluster assignments. The silhouette score is defined on [-1, 1].
# 
# A second method for determining the number of clusters to use is to look at the average distance of each point to its assigned cluster, also called a scree plot. This value approaches zero as the number of clusters goes to infinity, but with diminishing returns. A common metric is to look for a "bend" in the plot and set k equal to that.  
# 
# Below I will produce both plots.

# In[ ]:


# run silhouette score analysis for data projected onto 2 pc components
pca_comps = 2
pca = PCA(n_components=pca_comps)
x_pca = pca.fit_transform(X_tfidf.toarray())
comps = np.arange(2,10,1)
score = []
silhouette_avg = []
# get kmeans score versus number of components
for n_clusters in comps:
    kmeans = KMeans(n_clusters=n_clusters).fit(x_pca)
    score.append(kmeans.score(x_pca))
    cluster_labels = kmeans.predict(x_pca)
    silhouette_avg.append(silhouette_score(x_pca, cluster_labels))
    sample_silhouette_values = silhouette_samples(x_pca, cluster_labels)
    
    fig, ax1 = plt.subplots(1, 1)
    #ax0.plot(comps, score)
    #ax0.set_xlabel('# components')
    #ax0.set_title('k-means score')

    
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values =             sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        ax1.axvline(silhouette_avg[-1], linestyle='--', color='r')
        ax1.set_ylabel('Observation')
        ax1.set_title(f'Silhouette scores for k={i} clusters')
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
plt.subplots_adjust(hspace=1)
plt.show()

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
ax0.plot(comps, silhouette_avg)
ax0.set_xlabel('# k-means clusters')
ax0.set_ylabel('Mean silhouette score')
ax0.set_title(f'silhouette score: {pca_comps} PCs')
ax1.plot(comps, -np.array(score))
ax1.set_xlabel('# k-means clusters')
ax1.set_ylabel('Mean distance to cluster center')
ax1.set_title(f'Scree plot: {pca_comps} PCs')
plt.subplots_adjust(wspace=0.5)
plt.show()


# The silhouette score peaks at 4 clusters, and maybe if you squint you can convince yourself that there is a slight bend in the scree plot at k = 4. Choosing k is not an exact science and could also be informed by our prior knowledge of expected number of clusters. In this case, let's just try setting k equal to 4 and continuing.

# In[ ]:


# use 4 k-means clusters for data projected onto 2 PCs
k_clusts = 4
kmeans_pca = KMeans(n_clusters=k_clusts)

pca_comps = 2
pca = PCA(n_components=pca_comps)
x_pca = pca.fit_transform(X_tfidf.toarray())
kmeans_pca.fit(x_pca)

cmap = plt.cm.get_cmap(cmap_string, k_clusts)  

plot_pc_space(x_pca, pca, kmeans_pca.labels_, pca_comps, cmap)


# We see that K-means has essentially divided the observations based on physical location in PC space, which makes sense. Now let's generate word clouds for each cluster to see what topics each cluster roughly corresponds to.

# In[ ]:


wc, cl = assign_word_to_cluster(df.tags_single_string, kmeans_pca.labels_, countv.vocabulary_)
color_dict = assign_word_to_color(cl, cmap)

for i, clusters in enumerate(list(set(kmeans_pca.labels_))):
    cluster = np.where(kmeans_pca.labels_== clusters)
    text = ' '.join(df['tags_single_string'].str.lower().loc[cluster].values)
    wordcloud = WordCloud(max_font_size=50, max_words=50, background_color="black", collocations=False).generate(text)
    
    default_color = 'grey'
    grouped_color_func = SimpleGroupedColorFunc(color_dict, default_color)
    #grouped_color_func = GroupedColorFunc(color_dict, default_color)
    wordcloud.recolor(color_func=grouped_color_func)
    
    # Display the generated image
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(f'Cluster {i}')
    plt.show()


# In the word clouds above I have colored each word by the cluster in which it is most frequently found. We see that the cluster 0 (colored maroon), which resides in the bottom left region of our PC space plot, contains words like global, issues, and politics, which jibes with our understanding of PC 1 from earlier. Cluster 1 (peach) resides in the center of PC space and contains words like technology, design, and culture, very generic TED-type words. And clusters 2 (green) and 3 (purple) reside in the top and bottom right of PC space, respectively. As we expect from our understanding of PC space, cluster 2 contains science, and specifically biomedical, related words, and cluster 3 contains words pertaining to art, music and entertainment.
# 
# In the analysis above, we ran K-means on our data after projecting it down onto to only 2 PCs. Perhaps we can gain a more fine-grained understanding of talk clusters if we use more PCs. Now let's use 5 PCs and re-run our cluster number analysis to select a proper k.

# In[ ]:


# run silhouette score analysis for data projected onto 5 pc components
pca_comps = 5
pca = PCA(n_components=pca_comps)
x_pca = pca.fit_transform(X_tfidf.toarray())
comps = np.arange(2,10,1)
score = []
silhouette_avg = []
# get kmeans score versus number of components
for n_clusters in comps:
    kmeans = KMeans(n_clusters=n_clusters).fit(x_pca)
    score.append(kmeans.score(x_pca))
    cluster_labels = kmeans.predict(x_pca)
    silhouette_avg.append(silhouette_score(x_pca, cluster_labels))
    sample_silhouette_values = silhouette_samples(x_pca, cluster_labels)
    
    fig, ax1 = plt.subplots(1, 1)
    #ax0.plot(comps, score)
    #ax0.set_xlabel('# components')
    #ax0.set_title('k-means score')

    
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values =             sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        ax1.axvline(silhouette_avg[-1], linestyle='--', color='r')
        ax1.set_ylabel('Observation')
        ax1.set_title(f'Silhouette scores for k={i} clusters')
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
plt.subplots_adjust(hspace=1)
plt.show()

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
ax0.plot(comps, silhouette_avg)
ax0.set_xlabel('# k-means clusters')
ax0.set_ylabel('Mean silhouette score')
ax0.set_title(f'silhouette score: {pca_comps} PCs')
ax1.plot(comps, -np.array(score))
ax1.set_xlabel('# k-means clusters')
ax1.set_ylabel('Mean distance to cluster center')
ax1.set_title(f'Scree plot: {pca_comps} PCs')
plt.subplots_adjust(wspace=0.5)
plt.show()


# Here we see that 6 clusters maximizes the silhouette score. Let's continue with a k of 6. This should allow us to gain a more nuanced view of cluster identities.

# In[ ]:


# use 6 k-means clusters for data projected onto 5 PCs

k_clusts = 6
kmeans_pca = KMeans(n_clusters=k_clusts)

pca_comps = 5
pca = PCA(n_components=pca_comps)
x_pca = pca.fit_transform(X_tfidf.toarray())
kmeans_pca.fit(x_pca)

cmap = plt.cm.get_cmap(cmap_string, k_clusts)  

wc, cl = assign_word_to_cluster(df.tags_single_string, kmeans_pca.labels_, countv.vocabulary_)
color_dict = assign_word_to_color(cl, cmap)

for i, clusters in enumerate(list(set(kmeans_pca.labels_))):
    cluster = np.where(kmeans_pca.labels_== clusters)
    text = ' '.join(df['tags_single_string'].str.lower().loc[cluster].values)
    wordcloud = WordCloud(max_font_size=50, max_words=50, background_color="black", collocations=False).generate(text)
    
    default_color = 'grey'
    grouped_color_func = SimpleGroupedColorFunc(color_dict, default_color)
    #grouped_color_func = GroupedColorFunc(color_dict, default_color)
    wordcloud.recolor(color_func=grouped_color_func)
    
    # Display the generated image
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(f'Cluster {i}')
    plt.show()

plot_pc_space(x_pca, pca, kmeans_pca.labels_, pca_comps, cmap)


# Here we see that the central cluster, what I call the 'core Ted' cluster, has been broken up into 3 sub-groups, each of which leans toward one of the field-specific vertices but is perhaps slightly more general.

# K-means can assign new observations to a cluster, but it does not give us information about how likely assignment to each cluster is. This might be important especially in case like this where the data lie on a continuum, where we might want to know if an observation is almost equally likely to be a part of two adjacent clusters.
# 
# We can use a Gaussian mixture model (GMM) to once again assign each observation to a cluster while also enabling us to assign probabilities to each cluster assignment. Once again choosing the number of clusters is not trivial and there are semi-quantitative ways of doing that, such as using BIC and AIC, but for now I will just use 4 clusters for data projected onto 2 PCs, as the goal here is simply to illustrate that we can generate probabilities of cluster assignment.

# In[ ]:


from sklearn.mixture import GaussianMixture

# use 4 k-means clusters for data projected onto 2 PCs
n_clusts = 4
gm = GaussianMixture(n_components=n_clusts, covariance_type='diag')

pca_comps = 2
pca = PCA(n_components=pca_comps)
x_pca = pca.fit_transform(X_tfidf.toarray())
gm.fit(x_pca)
gm_labels = gm.predict(x_pca)

cmap = plt.cm.get_cmap(cmap_string, n_clusts)  

wc, cl = assign_word_to_cluster(df.tags_single_string, gm_labels, countv.vocabulary_)
color_dict = assign_word_to_color(cl, cmap)

for i, clusters in enumerate(list(set(gm_labels))):
    cluster = np.where(gm_labels == clusters)
    text = ' '.join(df['tags_single_string'].str.lower().loc[cluster].values)
    wordcloud = WordCloud(max_font_size=50, max_words=50, background_color="black", collocations=False).generate(text)
    
    default_color = 'grey'
    grouped_color_func = SimpleGroupedColorFunc(color_dict, default_color)
    #grouped_color_func = GroupedColorFunc(color_dict, default_color)
    wordcloud.recolor(color_func=grouped_color_func)
    
    # Display the generated image
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(f'Cluster {i}')
    plt.show()

plot_pc_space(x_pca, pca, gm_labels, pca_comps, cmap)


# We see the results are pretty similar to when we ran K-means using k=4. Let's use the probabilities available to use from our Gaussian mixture model to look at the words found only in talks whose cluster assignment has a probability greater than 0.75.

# In[ ]:


# let's only plot observations where the cluster assignment probability is greater than a threshold
p_min = 0.75
inds = np.where(np.max(gm.predict_proba(x_pca), axis=1) > p_min)
gm_labels_hi_prob = gm_labels[inds]
df_hi_prob = df.loc[inds]

for i, clusters in enumerate(list(set(gm_labels_hi_prob))):
    cluster = np.where(gm_labels_hi_prob == clusters)
    text = ' '.join(df_hi_prob['tags_single_string'].str.lower().iloc[cluster].values)
    wordcloud = WordCloud(max_font_size=50, max_words=50, background_color="black", collocations=False).generate(text)
    
    default_color = 'grey'
    grouped_color_func = SimpleGroupedColorFunc(color_dict, default_color)
    #grouped_color_func = GroupedColorFunc(color_dict, default_color)
    wordcloud.recolor(color_func=grouped_color_func)
    
    # Display the generated image
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(f'Cluster {i}')
    plt.show()
    

plot_pc_space(x_pca[inds], pca, gm_labels_hi_prob, pca_comps, cmap)


# Alternatively, we can look at talks whose cluster assignment has less than 0.75 probability.

# In[ ]:


# let's only plot observations where the cluster assignment probability is lower than a threshold
p_max = 0.75
inds = np.where(np.max(gm.predict_proba(x_pca), axis=1) < p_max)
gm_labels_low_prob = gm_labels[inds]
df_low_prob = df.loc[inds]

for i, clusters in enumerate(list(set(gm_labels_low_prob))):
    cluster = np.where(gm_labels_low_prob == clusters)
    text = ' '.join(df_low_prob['tags_single_string'].str.lower().iloc[cluster].values)
    wordcloud = WordCloud(max_font_size=50, max_words=50, background_color="black", collocations=False).generate(text)
    
    default_color = 'grey'
    grouped_color_func = SimpleGroupedColorFunc(color_dict, default_color)
    #grouped_color_func = GroupedColorFunc(color_dict, default_color)
    wordcloud.recolor(color_func=grouped_color_func)
    
    # Display the generated image
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(f'Cluster {i}')
    plt.show()
    
plot_pc_space(x_pca[inds], pca, gm_labels_low_prob, pca_comps, cmap)


# We see that the talks that are less confidently assigned a cluster lie towards the center of PC space, and their word clouds are perhaps slightly more generic than the more confidently classified talks. Being able to assign a probability to each cluster could also be useful for generating primary and secondary labels for each data point.

# Finally, we can also use Non-negative matrix factorization to group talks. NMF is sometimes used for 'topic modeling' or assigning documents to discrete topics, which is exactly what we have been exploring in this notebook. NMF will enable us to cluster observations as well as directly observe which words contributed to the cluster assignment. We again have to supply the number of components or topics for NMF to find, and here I will choose 5, though in theory this could be chosen in a more principled manner by combining metrics and domain knowledge.

# In[ ]:


from sklearn.decomposition import NMF

n_hypothesized_topics = 5
nmf = NMF(n_components=n_hypothesized_topics, random_state=42)
nmf.fit(X_tfidf)

top_words_to_grab = 15
cmap = plt.cm.get_cmap(cmap_string, n_hypothesized_topics)  

feature_names = countv.get_feature_names()

fig, ax = plt.subplots(n_hypothesized_topics, 1, figsize=(4, 20))
for i, comp in enumerate(nmf.components_):
    top_cluster_words = [feature_names[ind] for ind in comp.argsort()[-1:-top_words_to_grab:-1]]
    bp = ax[i].barh(top_cluster_words, comp[comp.argsort()[-1:-top_words_to_grab:-1]], 
              color=cmap(i), edgecolor='black')
    ax[i].set_title(f'Top word loadings for cluster {i}')
    
    ax[i].invert_yaxis()

plt.show()
nfm_labels = nmf.transform(X_tfidf).argmax(axis=1)


# In[ ]:


wc, cl = assign_word_to_cluster(df.tags_single_string, nfm_labels, countv.vocabulary_)
color_dict = assign_word_to_color(cl, cmap)

for i, clusters in enumerate(list(set(nfm_labels))):
    cluster = np.where(nfm_labels == clusters)
    text = ' '.join(df['tags_single_string'].str.lower().loc[cluster].values)
    wordcloud = WordCloud(max_font_size=50, max_words=50, background_color="black", collocations=False).generate(text)
    
    default_color = 'grey'
    grouped_color_func = SimpleGroupedColorFunc(color_dict, default_color)
    #grouped_color_func = GroupedColorFunc(color_dict, default_color)
    wordcloud.recolor(color_func=grouped_color_func)
    
    # Display the generated image
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(f'Cluster {i}')
    plt.show()

    
pca_comps = 2
pca = PCA(n_components=pca_comps)
x_pca = pca.fit_transform(X_tfidf.toarray())
plot_pc_space(x_pca, pca, nfm_labels, pca_comps, cmap)


# We know the broad category that each of these clusters is representing, so let's name them and add a new column to our original dataframe to generate a high-level tag for each talk.

# In[ ]:


cluster_names = {
    0: 'global issues and poverty',
    1: 'core TED',
    2: 'scientific research',
    3: 'art',
    4: 'society',
}

labels_to_cluster_names = [cluster_names[x] for x in nfm_labels]
df['general_category'] = labels_to_cluster_names


# Now that we have done all this unsupervised analysis to group talks into high-level categories, we may be curious about whether there are any differences in engagement between these categories.
# 
# Let's groupby our new high-level categories and plot how many times each category has been viewed on average.

# In[ ]:


ax = df['views'].groupby(df.general_category).agg(['mean', 'sem']).plot(kind='barh', xerr='sem')
ax.set_ylabel('General Category')
ax.set_xlabel('Total Views')
ax.get_legend().remove()
plt.show()


# Of course, number of views depends on how long the video has been available. Let's add a new column to the dataframe to get the age of the video and then normalize views by the age.

# In[ ]:


from datetime import datetime

df['video_age_in_days'] = (datetime.now()-df.date).astype('timedelta64[D]')
df['views_per_day'] = df['views'] / df['video_age_in_days']


# And let's plot the normalized views now for each group:

# In[ ]:


ax = df['views_per_day'].groupby(df.general_category).agg(['mean', 'sem']).plot(kind='barh', xerr='sem')
ax.set_ylabel('General Category')
ax.set_xlabel('Views per day')
ax.get_legend().remove()
plt.show()


# There you have it. It seems talks focusing on societal issues are viewed the most, while those focusing on global issues and poverty are viewed less.
# 
# One challenge in data analysis is that there are many ways of doing similar things, with some methods simpler, others more powerful, and others more tailored to a specific use case. Latent Dirichlet Allocation (LDA) can use be used for topic modeling, and powerful libraries exist for advanced NLP and part-of-speech tagging. Nevertheless, I hope this notebook has introduced some simple ways you can analyze text in an unsupervised manner using only sklearn. 
