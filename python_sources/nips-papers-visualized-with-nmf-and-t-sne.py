#!/usr/bin/env python
# coding: utf-8

# # NIPS 2D: All NIPS papers visualized on a 2D map
# ## Primary techniques used: NMF for topic modeling and t-SNE for 2D-embedding
# 
# This is a little test how topic modeling works on visualizing the content of all NIPS papers until 2016. 2D-embeddings are always easy to understand for us, therefore I will try to create a scatterplot that captures both the content similarity of papers as well as their distinct topics. This notebook uses the papers' textual content, not the titles or something else. 
# 
# The notebook is rather short and divided into two parts:
# 
# - data loading and topic modeling
# - visualization with t-SNE

# ### Part 1: data loading and topic modeling

# In[ ]:


#data wrangling packages
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import random 
random.seed(13)

#visualization packages
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


df = pd.read_csv("../input/papers.csv")
df.head()


# In[ ]:


len(df)


# In[ ]:


n_features = 1000
n_topics = 8
n_top_words = 10


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,max_features=n_features,stop_words='english')

tfidf = tfidf_vectorizer.fit_transform(df['paper_text'])


nmf = NMF(n_components=n_topics, random_state=0,alpha=.1, l1_ratio=.5).fit(tfidf)

print("Topics found via NMF:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)


# In the snippet above you get a first glimpse about the topics found. I honestly don't always find the top words useful when used alone. Another approach I like a lot: looking in which papers the topics are "activated" the most. We therefore transform the tfidf-matrix into the nmf-embedding and have a look with np.argsort, which research papers have the strongest link to each topic.

# In[ ]:


nmf_embedding = nmf.transform(tfidf)
nmf_embedding = (nmf_embedding - nmf_embedding.mean(axis=0))/nmf_embedding.std(axis=0)


# In[ ]:


top_idx = np.argsort(nmf_embedding,axis=0)[-3:]

count = 0
for idxs in top_idx.T: 
    print("\nTopic {}:".format(count))
    for idx in idxs:
        print(df.iloc[idx]['title'])
    count += 1


# Looks much better to me. 
# 
# The next part is very subjective. But looking at those titles and top-words I decide to assign the following descriptions to the topics: 
# 
# - **topic 0:** optimization algorithms
# - **topic 1:** neural network application
# - **topic 2:** reinforcement learning
# - **topic 3:** bayesian methods
# - **topic 4:** image recognition
# - **topic 5:** artificial neuron design
# - **topic 6:** graph theory
# - **topic 7:** kernel methods

# In[ ]:


topics = ['optimization algorithms',
          'neural network application',
          'reinforcement learning',
          'bayesian methods',
          'image recognition',
          'artificial neuron design',
          'graph theory',
          'kernel methods'
         ]


# ### Part 2: visualize the findings
# Like lots of people in the data science community I'm a huge fan of t-SNE. It's capabilities to visualize complex relationships are stunning. 

# In[ ]:


tsne = TSNE(random_state=3211)
tsne_embedding = tsne.fit_transform(nmf_embedding)
tsne_embedding = pd.DataFrame(tsne_embedding,columns=['x','y'])
tsne_embedding['hue'] = nmf_embedding.argmax(axis=1)


# The next cell is a little bit hacky. Since I want to create a custom legend afterwards I first plot a dummy output and then obtain the rgb-values used in this plot. Afterwards I have a hard-coded color list I can use for the custom legend. I honestly didn't find a better way. Since I don't want to output the plot in this notebook (it's boring and would be the one published) I didn't find a better way. 

# In[ ]:


###code used to create the first plot for getting the colors 
#plt.style.use('ggplot')

#fig, axs = plt.subplots(1,1, figsize=(5, 5), facecolor='w', edgecolor='k')
#fig.subplots_adjust(hspace = .1, wspace=.001)

#legend_list = []

#data = tsne_embedding
#scatter = plt.scatter(data=data,x='x',y='y',s=6,c=data['hue'],cmap="Set1")
#plt.axis('off')
#plt.show()

#colors = []
#for i in range(len(topics)):
#    idx = np.where(data['hue']==i)[0][0]
#    color = scatter.get_facecolors()[idx]
#    colors.append(color)
#    legend_list.append(mpatches.Ellipse((0, 0), 1, 1, fc=color))
 
colors = np.array([[ 0.89411765,  0.10196079,  0.10980392,  1. ],
 [ 0.22685121,  0.51898501,  0.66574396,  1. ],
 [ 0.38731259,  0.57588621,  0.39148022,  1. ],
 [ 0.7655671 ,  0.38651289,  0.37099578,  1. ],
 [ 1.        ,  0.78937332,  0.11607843,  1. ],
 [ 0.75226453,  0.52958094,  0.16938101,  1. ],
 [ 0.92752019,  0.48406   ,  0.67238756,  1. ],
 [ 0.60000002,  0.60000002,  0.60000002,  1. ]])

legend_list = []

for i in range(len(topics)):   
    color = colors[i]
    legend_list.append(mpatches.Ellipse((0, 0), 1, 1, fc=color))


# Lets try to get some insights out of the data. I think it would be interesting to visualize how the topics at NIPS emerged over time. Let's do this!

# In[ ]:


matplotlib.rc('font',family='monospace')
plt.style.use('ggplot')


fig, axs = plt.subplots(3,2, figsize=(10, 15), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .1, wspace=0)

axs = axs.ravel()

count = 0
legend = []
for year, idx in zip([1991,1996,2001,2006,2011,2016], range(6)):
    data = tsne_embedding[df['year']<=year]
    scatter = axs[idx].scatter(data=data,x='x',y='y',s=6,c=data['hue'],cmap="Set1")
    axs[idx].set_title('published until {}'.format(year),**{'fontsize':'10'})
    axs[idx].axis('off')

plt.suptitle("all NIPS proceedings clustered by topic",**{'fontsize':'14','weight':'bold'})
plt.figtext(.51,0.95,'unsupervised topic modeling with NMF based on textual content + 2D-embedding with t-SNE:', **{'fontsize':'10','weight':'light'}, ha='center')


fig.legend(legend_list,topics,loc=(0.1,0.89),ncol=3)
plt.subplots_adjust(top=0.85)

plt.show()


# For me plots like this can be very helpful. What can we see here for example? Did you realize that neural networks were the most active topic at the eary times of NIPS? In fact you can cleary see that the blue cluster (neural networks) was strongly present right from the beginning. 
# 
# Also interesting for me is how kernel methods took over NIPS in the last years. What else can you see in this plot? 
# 
# Of course, there is lots of room for improvement of the code. Here are just some examples:
# 
# - **Topic count:** I chose 8 by trial and error. It would be worth a look trying out several topic numbers.
# - **Topic description:** in this code I only looked at the top-10 words and top-3 titles (eventough I also looked into the top-10 titles as well, which didn't change anything)
# - **NMF parameters:** the parameters I used where standard ones from sklearn tutorials. Why not try different ones here
# 
# It's also important to note that topic modeling doesn't create one-hot encodings. Eventough we assign each paper to its most activated topic in this notebook, they are of course represented by mixtures of more than one topic. 
