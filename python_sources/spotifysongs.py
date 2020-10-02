#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install pandasql


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from pandasql import sqldf
from scipy.stats import ttest_ind
from scipy.stats.stats import pearsonr
pysqldf = lambda q: sqldf(q, globals())
from sklearn.manifold import TSNE
import time
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import seaborn as sns
import matplotlib.patheffects as PathEffects
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
RS = 123

get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv("../input/top-spotify-songs-from-20102019-by-year/top10s.csv", encoding='ISO-8859-1')
data.head(5)


# ### Possible Analyses
# ---
# 1. Genre
# 2. How BPM affects the Popularity
# 3. How BPM has changed over the decade
# 4. Relationship between Genre and BPM
# 5. Relationship between Loudness and Popularity
# 6. Have the song become louder over the decade
# 7. Most Consistent Artist
# 8. Most popular artist
# 9. Correlation between loudness and Energy and BPM and energy and popularity
# 10. Spch and popularity, spch and artist
# 11. Topic of the song(Depends on title using NLP)
# 12. cluster the songs using topic, genre, artist bpm-spch
# ---
# More can be added later on!

# In[ ]:


print(pysqldf("""select year, count(distinct title) from data
group by 1
order by 2"""))


# In[ ]:


print("top genre on spotify's top 50 chart is")
print(pysqldf("select `top genre`, count(*) as cnt from data                     group by 1                     order by 2 desc").head())


# ### The most common genre on the chart is Dance pop which contains roughly 50% of the songs on the chart

# In[ ]:


data_filter = pysqldf("""select artist, year, count(distinct title) as cnt from data
                      group by 1,2 
                      order by 2,1""")
data_filter = pd.pivot_table(data_filter, index = 'artist', columns = 'year', 
                   values = 'cnt', aggfunc = np.sum, fill_value = 0)
data_filter['total_songs'] = data_filter.sum(axis = 1)
data_filter['mean_songs'] = data_filter.loc[:,data_filter.columns != 'total_songs'].mean(axis = 1)
data_filter.head()


# In[ ]:


data_filter.loc[data_filter['mean_songs'] > 0.5,['mean_songs']].sort_values('mean_songs', ascending = False).plot(kind = 'bar')


# ### Katy Perry has been the most popular artist over the decade. Atleast 1.6 songs created by Katy Perry have charted on the Top 50 list over the decade.
# 
# ### Is there a trend in her popularity?

# In[ ]:


cols = list(set(data_filter.columns) - set( ['artist','year','total_songs', 'mean_songs']))
cols = sorted(cols)
y = data_filter.reset_index()[data_filter.reset_index()['artist'] == 'Katy Perry'][cols]
plt.scatter(x = cols, y = y)
print(y)


# ### Last two years of the decade were not really great for Katy Perry, non of her songs charted on the top 50 list of the year.

# In[ ]:


data_filter = pysqldf("""select year, avg(dB) as avg,max(dB) as max, min(dB) as min 
from data group by 1
order by 1""")

data_filter.set_index('year')[['avg']].plot()


# In[ ]:


data[['dB', 'year']].boxplot(by = 'year')


# In[ ]:


print(pysqldf("""select title,artist,`top genre` from data
where dB < -10"""))


# In[ ]:


data_filter = pysqldf("""select `top genre`, avg(dB) from data
group by 1
order by 2""")
data_filter.set_index('top genre').plot(kind = "bar", figsize = (20,10))


# In[ ]:


corr, p = pearsonr(x = data['dB'], y = data['pop'])
print("Correlation coefficient is %0.4f with a p-value of %0.2f" %(corr, p))


# In[ ]:


data_filter = pysqldf("""select `top genre`, avg(dB), -avg(pop) from data
group by 1
order by 2""")
data_filter.set_index('top genre').plot(kind = "bar", figsize = (20,10))


# ### * Upon listening to a few escape room songs, I did feel they were louder that other genres. You can sample a few songs at [Every noise](http://everynoise.com)
# ### * Over the years the songs are getting mellower on an average
# ### * Loudness of a song has a mild correlation to the popularity, meaning loudness only slightly affects the popularity.

# In[ ]:


print(pysqldf("""select title, artist from data where `top genre` = 'escape room'"""))


# In[ ]:


data_filter = data[(data['nrgy'] > 20) & (data['dnce'] > 20)]
plt.scatter(x = data_filter['nrgy'], y = data_filter['dnce'])
corr, p = pearsonr(x = data_filter['nrgy'], y = data_filter['dnce'])
print("Correlation coefficient is %0.4f with a p-value of %0.2f" %(corr, p))
print("We can say that there's very little relation between energy of the song and how groovy it is!")


# In[ ]:


dance_pop = data[data['top genre'] == 'dance pop']
others = data[data['top genre'] != 'dance pop']
x = dance_pop['dnce']
y = others['dnce']
plt.hist(x, bins = 30, alpha=0.7, label='dance_pop')
plt.hist(y, bins = 30, alpha=0.3, label='others')
plt.legend(loc='upper right')
plt.show()


# In[ ]:


stat, p = ttest_ind(a = dance_pop['dnce'], b = others['dnce'], equal_var=False)
print("p-value is %0.3f" %p)


# In[ ]:


cols = ['nrgy', 'bpm', 'dnce', 'pop']
pp = sns.pairplot(data[cols], size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Pairwise Plots', fontsize=14)


# In[ ]:


for col in cols:
    q = "select year, avg(%s) as avg_%s from data "%(col,col)
    q+= "group by 1 order by 1"
#     print(q)
    data_filter = pysqldf(q)
    q = "avg_%s"%(col)
    data_filter.set_index('year')[[q]].plot()
    plt.show()


# ### * The dance_pop very groovy genre is very different from other songs on the chart, in terms of the song's danceability. as we can see from the p-value(t-test for a large sample size)
# ### * One observation, all the songs are very similiar in nature to each other. We can probably verify this claim later on by running a K-Means clustering
# ### * a very weird trend is coming out from this dataset, danceability is increasing over the year even though the energy of the songs show a downwards trend
# ### * A thing to notice here is: Average popularity has been increasing. A possible explanation for such a trend is the popularity value is being constantly updated and the data shows the current popularity of the song rather than it's peak popularity

# ### Using Doc2Vec model from gensim we convert the title to a vector of length 10. For this we first train the model using the corpus of title that we already have.

# In[ ]:


documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data['title'])]


# In[ ]:


documents[1]


# In[ ]:


model = Doc2Vec(documents, vector_size=10, window=2, min_count=1, workers=4)


# In[ ]:


vector = model.infer_vector(["Love The Way You Lie"])


# In[ ]:


vector


# In[ ]:


vector = [model.infer_vector([i]) for i in list(data['title'])]
np.array(vector).shape


# ### To visualize the similarity between the songs on the chart, we'll leverage T-SNE algorithm. But for this we need to find cluster

# In[ ]:


import pandas as pd
# from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
pop = pd.qcut(data['pop'], q= 10, labels = False)
data_filter = data.drop(['artist', 'title','Unnamed: 0', 'pop'], axis = 1)
data_filter = pd.get_dummies(data_filter, prefix = 'top_genre', columns = ['top genre'], drop_first= True)
data_filter = data_filter.to_numpy()

sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data_filter)
    clusters = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# ### We will cluster the data into two groups as the elbow is formed at 2 also we drop popularity index for clustering as popularity is updated regularly!

# In[ ]:


# pop = pd.qcut(data['pop'], q= 10, labels = False)
data_filter = data.drop(['artist', 'title','Unnamed: 0', 'pop'], axis = 1)
data_filter = pd.get_dummies(data_filter, prefix = 'top_genre', columns = ['top genre'], drop_first= True)
# year = data_filter['year']
# data_filter = data_filter.drop(['year'], axis = 1)
data_filter = data_filter.to_numpy()
kmeans = KMeans(n_clusters=2, max_iter=1000).fit(data_filter)
clusters = kmeans.labels_
# data_filter


# In[ ]:


data_filter = np.concatenate((data_filter,vector), axis = 1)
data_filter.shape


# In[ ]:


# Utility function to visualize the outputs of PCA and t-SNE

def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


# In[ ]:


time_start = time.time()

fashion_tsne = TSNE(perplexity = 10, learning_rate = 100, n_iter = 5000, random_state=RS).fit_transform(data_filter)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


# In[ ]:


fashion_tsne.shape


# In[ ]:


fashion_scatter(fashion_tsne, clusters)

