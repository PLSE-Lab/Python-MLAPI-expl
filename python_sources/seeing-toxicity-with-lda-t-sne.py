#!/usr/bin/env python
# coding: utf-8

# I wanted to see if we could visualize how a machine learning model classifies text data. I use Latent Dirichlet Allocation(LDA) for topic modeling along with T-distributed Stochastic Neighbor Embedding (t-SNE) to provide a 2-d picture. Although this method won't get the highest AUC possible, it does a reasonably good job of providing insight into text classification.
# 
# I've found t-SNE to be a great tool in helping non-data scientists understand the capabilities and limits of numerical models. Once you get people comfortable with the idea of condensing numerous factors into 2 or 3 dimensions, it seems easier for them to identify with the spatial map produced by t-SNE.
# 

# In[ ]:


# %autosave 600
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
# from MulticoreTSNE import MulticoreTSNE as TSNE
# %load_ext wurlitzer


# I won't distinguish between the various categories of toxicity. We'll mark comments from any of the six categories. To make it a little more interesting, our target variable will show the number of categories observed for each comment. For example, a comment classified as toxic, severe_toxic and obscene gets a 3. I'll color the  t-SNE plot to highlight comments falling in 1 or more categories.
# 
# I took 10% of the data to reduce the clutter of the picture and speed things up. Running the whole train set shows similar results.

# In[ ]:


train = pd.read_csv('../input/train.csv').fillna(' ')

trainX = train['comment_text']
target = train.sum(axis=1).values

sss = StratifiedShuffleSplit(n_splits=5, train_size=0.10)
for train_index, test_index in sss.split(trainX, target):
    train_text = trainX.iloc[train_index] 
    train_tgt = target[train_index]


# Now we can count the word features and run LDA. I chose parameters to count the top 5000 words (per tfi/idf) present in each comment and then represent the comment as a vector of 16 topics. 

# In[ ]:


maxfeats = 5000
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
   # token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=maxfeats)
word_vectorizer.fit(train_text)
train_features = word_vectorizer.transform(train_text)

classifier = LatentDirichletAllocation(n_components=16, learning_method=None, n_jobs=3, verbose=1)
train_lda = classifier.fit_transform(train_features, train_tgt)
train_lda.shape


# Next we can represent the topic vectors in 2-d with t-SNE. Finally we plot the results with plotly.

# In[ ]:


####
#train_lda = train_features.toarray()

tsne = TSNE(n_components=2, perplexity=8, n_iter=1600, verbose=1, angle=0.5)
train_tsne = tsne.fit_transform(train_lda)
x_tsne = train_tsne[:, 0]
y_tsne = train_tsne[:, 1]


# In[ ]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Scatter, Figure, Layout
init_notebook_mode(connected=False)

# create datafRAME WITH comments, target, tsnex,tsney
#separate into 2 groups of x_nice, x_notnice, y_nice, y_notnice
plotme = pd.DataFrame({'comment':train_text, 'class':train_tgt, 'xcoord': x_tsne, 'ycoord':y_tsne})
nices = plotme[plotme['class'] == 0]
notnices = plotme[plotme['class'] > 0]


# In[ ]:



trace_nices = Scatter(
    x = nices['xcoord'],
    y = nices['ycoord'],
    mode = 'markers',
    marker = dict(
      size=7,
      color='lightgray',
      symbol='circle',
      line = dict(width = 0,
        color='gray'),
      opacity = 0.3
     ),
    text=nices['comment']
)

trace_notnices = Scatter(
    x = notnices['xcoord'],
    y = notnices['ycoord'],
    mode = 'markers',
    marker = dict(
      size=8,
      color=notnices['class'],
      symbol='triangle-up',
      line = dict(width = 0,
        color='Darkred'),
      opacity = 0.6
     ),
    text=notnices['comment']
)

data=[trace_nices, trace_notnices]

layout = Layout(
    title = 'We See You...',
    showlegend=False,
    xaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=False
    ),
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=False
    )
)
# Plot and embed in ipython notebook!
fig = Figure(data=data, layout=layout)
iplot(fig, filename='jupyter/scatter1')


# You can see the concentration of toxic comments in one region of the plot. You can also see that many comments classified as toxic are scattered throughout the space. It gives some added context to a numerical score or a confusion matrix. And it shows those less familiar with machine learning how the model isn't perfect, but it may be useful, which is true of any model.
# 
# With the interactive plot you can zoom in on regions with the plotly tools and hover over points to read the comments. It's a great way to give others visibility into the problem anad a potential solution!
