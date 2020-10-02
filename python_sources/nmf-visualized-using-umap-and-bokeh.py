#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# Bokeh
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, CustomJS, ColumnDataSource, Slider, Range1d
from bokeh.layouts import column
from bokeh.palettes import all_palettes
output_notebook()


# ## 1. Loading data
# We load docs from [NIPS Papers](https://www.kaggle.com/benhamner/nips-papers) dataset.

# In[ ]:


df = pd.read_csv("../input/nips-papers/papers.csv")
print(df.paper_text[0][:500] + ' ...')


# ## 2. Lemmatization
# 
# Apply lemmatization `spaCy` [framework](https://spacy.io/). **Lemmatization** is the redusing a word to its "dictionary form" (word's *lemma*). 

# In[ ]:


get_ipython().run_cell_magic('time', '', "import spacy\n\nnlp = spacy.load('en', disable=['parser', 'ner'])\ndf['paper_text_lemma'] = df.paper_text.map(lambda x: [token.lemma_ for token in nlp(x) if token.lemma_ != '-PRON-' and token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'}])\n\n# Final cleaning\ndf['paper_text_lemma'] = df.paper_text_lemma.map(lambda x: [t for t in x if len(t) > 1])\n\n# Example\nprint(df['paper_text_lemma'][0][:25], end='\\n\\n')")


# ## 3. TFIDF and UMAP
# 
# Constructing [TFIDF-matrix](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).

# In[ ]:


get_ipython().run_cell_magic('time', '', "from sklearn.feature_extraction.text import TfidfVectorizer\n\nnp.random.seed(42)\nn_features=2000\ntfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features, ngram_range=(1,2), stop_words='english')\ntfidf = tfidf_vectorizer.fit_transform(df.paper_text_lemma.map(lambda x: ' '.join(x)))")


# Now, we'll build an embedding of our `n_feature`-dimesional space into 2D using `UMAP` ([Uniform Manifold Approximation and Projection for Dimension Reduction](https://umap-learn.readthedocs.io/en/latest/)) packege for visualization.

# In[ ]:


get_ipython().run_cell_magic('time', '', "import umap\n\numap_embr = umap.UMAP(n_neighbors=10, metric='cosine', min_dist=0.1, random_state=42)\nembedding = umap_embr.fit_transform(tfidf.todense())\nembedding = pd.DataFrame(embedding, columns=['x','y'])")


# So, let's see what we have...

# In[ ]:


source = ColumnDataSource(
        data=dict(
            x = embedding.x,
            y = embedding.y,
            title = df.title,
            year = df.year,
        )
    )
hover_emb = HoverTool(names=["df"], tooltips="""
    <div style="margin: 10">
        <div style="margin: 0 auto; width:300px;">
            <span style="font-size: 12px; font-weight: bold;">Title:</span>
            <span style="font-size: 12px">@title</span>
            <span style="font-size: 12px; font-weight: bold;">Year:</span>
            <span style="font-size: 12px">@year</span>
        </div>
    </div>
    """)
tools_emb = [hover_emb, 'pan', 'wheel_zoom', 'reset']
plot_emb = figure(plot_width=600, plot_height=600, tools=tools_emb, title='Papers')
plot_emb.circle('x', 'y', size=5, fill_color='green',
                alpha=0.7, line_alpha=0, line_width=0.01, source=source, name="df")

plot_emb.x_range = Range1d(-8, 6)
plot_emb.y_range = Range1d(-8, 7)

layout = column(plot_emb)
show(layout)


# It seems we have some structure there, let's investigate further.

# ## 4. Gensim NMF model and Coherence
# 
# Let's organize the text into a datastructure sutable for `gensim` [non-negative matrix factorization](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization) model.

# In[ ]:


get_ipython().run_cell_magic('time', '', "from gensim import corpora, models\nnp.random.seed(42)\n\n# Create a corpus from a list of texts\ntexts = df['paper_text_lemma'].values\ndictionary = corpora.Dictionary(texts, prune_at=2000)\ncorpus = [dictionary.doc2bow(text) for text in texts]")


# Training the `NMF` models. Here, we'll train approximately $50$ models (for the numbers of topics (`n_topics`) between $3$ and $50$). For each model we calculate the *coherence score* (coherence score is cculated for each topic within a particular module). All those scores will be saved into `coh_list` (a list of coherence scores for every model). For example, the first element of the list is a list consisting of $3$ scores (since the first model will have only $3$ topics. The second element is the list of length $4$, and so on.
# 
# We are using the coherence metric called `UMass` (aka *intrinsic measure*).

# In[ ]:


get_ipython().run_cell_magic('time', '', "from gensim.models.nmf import Nmf\nfrom gensim.models.coherencemodel import CoherenceModel\n\ncoh_list = []\nfor n_topics in range(3,50+1):\n    # Train the model on the corpus\n    nmf = Nmf(corpus, num_topics=n_topics, id2word=dictionary, random_state=42)\n    # Estimate coherence\n    cm = CoherenceModel(model=nmf, texts=texts, dictionary=dictionary, coherence='u_mass')\n    coherence = cm.get_coherence_per_topic() # get coherence value\n    coh_list.append(coherence)")


# Let plot the coherence scores and guess the number of topics. First, we calculate mean score and the standard deviation for each model. The blue line shows the means and the green region represents the standard deviations.

# In[ ]:


# Coherence scores:
coh_means = np.array([np.mean(l) for l in coh_list])
coh_stds = np.array([np.std(l) for l in coh_list])

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.xticks(np.arange(3, 50+1, 3.0));
plt.plot(range(3,50+1), coh_means);
plt.fill_between(range(3,50+1), coh_means-coh_stds, coh_means+coh_stds, color='g', alpha=0.05);
plt.vlines([6, 12, 23], -1.1, 0, color='red', linestyles='dashed',  linewidth=1);
plt.hlines([-0.645], 3, 50, color='black', linestyles='dotted',  linewidth=0.5);
plt.ylim(-1.1,0);


# As "good candidates" for the number of topics we'll chose a few local minima of the graph. Those are `n_topic=6` and `n_topic=12`. Also, the mean coherence plot seems to have a starting plato around `n_topic=23`. We'll investigate those values below.

# ## 5. NMF models in details

# ### 5.1 NMF-6

# #### 5.1.1. Topics
# 
# For further investigation we'll use NMF algorithm from another packege (`NMF` from `sklearn.decomposition`).

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.decomposition import NMF\n\nn_topics=6\nn_top_words = 15\nnmf = NMF(n_components=n_topics, random_state=42, alpha=.1, l1_ratio=.5).fit(tfidf)\nnmf_embedding = nmf.transform(tfidf)\nfeature_names = tfidf_vectorizer.get_feature_names()\nprint("Topics found via NMF:")\nfor topic_idx, topic in enumerate(nmf.components_):\n    print("\\nTopic {}:".format(topic_idx))\n    print(" ".join([\'[{}]\'.format(feature_names[i]) for i in topic.argsort()[:-n_top_words - 1:-1]]))\nprint()')


# In[ ]:


topics = ['Optimization Algorithms',
          'Artificial Neurons',
          'Game Theory/Reinf. Learn.',
          'Neural Networks',
          'Bayesian Methods',
          'Kernel Methods'          
         ]


# #### 5.1.2. Bokeh interactive plot

# In[ ]:


centroids = umap_embr.transform(nmf.components_)
embedding['hue'] = nmf_embedding.argmax(axis=1)
my_colors = [all_palettes['Category20'][20][i] for i in embedding.hue]
source = ColumnDataSource(
        data=dict(
            x = embedding.x,
            y = embedding.y,
            colors = my_colors,
            topic = [topics[i] for i in embedding.hue],
            title = df.title,
            year = df.year,
            alpha = [0.7] * embedding.shape[0],
            size = [7] * embedding.shape[0]
        )
    )
hover_emb = HoverTool(names=["df"], tooltips="""
    <div style="margin: 10">
        <div style="margin: 0 auto; width:300px;">
            <span style="font-size: 12px; font-weight: bold;">Topic:</span>
            <span style="font-size: 12px">@topic</span>
            <span style="font-size: 12px; font-weight: bold;">Title:</span>
            <span style="font-size: 12px">@title</span>
            <span style="font-size: 12px; font-weight: bold;">Year:</span>
            <span style="font-size: 12px">@year</span>
        </div>
    </div>
    """)
tools_emb = [hover_emb, 'pan', 'wheel_zoom', 'reset']
plot_emb = figure(plot_width=700, plot_height=700, tools=tools_emb, title='Papers')
plot_emb.circle('x', 'y', size='size', fill_color='colors', 
                 alpha='alpha', line_alpha=0, line_width=0.01, source=source, name="df", legend='topic')

for i in range(n_topics):
    plot_emb.cross(x=centroids[i,0], y=centroids[i,1], size=15, color='black', line_width=2, angle=0.79)
plot_emb.legend.location = "bottom_left"
plot_emb.legend.label_text_font_size= "8pt"
plot_emb.legend.spacing = -5
plot_emb.x_range = Range1d(-9, 7)
plot_emb.y_range = Range1d(-9, 7)

callback = CustomJS(args=dict(source=source), code=
    """
    var data = source.data;
    var f = cb_obj.value
    x = data['x']
    y = data['y']
    colors = data['colors']
    alpha = data['alpha']
    title = data['title']
    year = data['year']
    size = data['size']
    for (i = 0; i < x.length; i++) {
        if (year[i] <= f) {
            alpha[i] = 0.9
            size[i] = 7
        } else {
            alpha[i] = 0.05
            size[i] = 4
        }
    }
    source.change.emit();
    """)

slider = Slider(start=df.year.min()-1, end=df.year.max(), value=2016, step=1, title="Before year")
slider.js_on_change('value', callback)

layout = column(slider, plot_emb)
show(layout)


# #### 5.1.3. Static Picture

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
get_ipython().run_line_magic('matplotlib', 'inline')

legend_list = []
for color in all_palettes['Category20'][20][:n_topics]:   
    legend_list.append(mpatches.Ellipse((0, 0), 1, 1, fc=color))
    
fig,ax = plt.subplots(figsize=(12,13))
ax.scatter(embedding.x, embedding.y, c=my_colors, alpha=0.7)
ax.scatter(centroids[:,0], centroids[:,1], c='black', s=100, alpha=0.7, marker='x')
ax.set_title('6 topics found via NMF');
fig.legend(legend_list, topics, loc=(0.18,0.87), ncol=3)
plt.subplots_adjust(top=0.82)
plt.suptitle("NIPS clustered by topic", **{'fontsize':'14','weight':'bold'});
plt.figtext(.51,0.95, 'topic modeling with NMF + 2D-embedding with UMAP', 
            **{'fontsize':'12','weight':'light'}, ha='center');


# ### 5.2. NMF-12

# #### 5.2.1. Topics

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.decomposition import NMF\nn_topics=12\nn_top_words = 15\nnmf = NMF(n_components=n_topics, random_state=42, alpha=.1, l1_ratio=.5).fit(tfidf)\nnmf_embedding = nmf.transform(tfidf)\nfeature_names = tfidf_vectorizer.get_feature_names()\nprint("Topics found via NMF:")\nfor topic_idx, topic in enumerate(nmf.components_):\n    print("\\nTopic {}:".format(topic_idx))\n    print(" ".join([\'[{}]\'.format(feature_names[i]) for i in topic.argsort()[:-n_top_words - 1:-1]]))\nprint()')


# In[ ]:


topics = ['Optimization Algorithms',
          'Neural Networks',
          'Reinforcement Learning',
          'Image Recognition',
          'Bayesian Methods',
          'Visual Neurons',
          'Graph/Tree Methods',
          'Classification Problems',
          'Kernel Methods',
          'Clastering Methods',
          'Game Theory',
          'Artificial Neurons'
         ]


# #### 5.2.2. Bokeh interactive plot

# In[ ]:


centroids = umap_embr.transform(nmf.components_)
embedding['hue'] = nmf_embedding.argmax(axis=1)
my_colors = [all_palettes['Category20'][20][i] for i in embedding.hue]
source = ColumnDataSource(
        data=dict(
            x = embedding.x,
            y = embedding.y,
            colors = my_colors,
            topic = [topics[i] for i in embedding.hue],
            title = df.title,
            year = df.year,
            alpha = [0.7] * embedding.shape[0],
            size = [7] * embedding.shape[0]
        )
    )
hover_emb = HoverTool(names=["df"], tooltips="""
    <div style="margin: 10">
        <div style="margin: 0 auto; width:300px;">
            <span style="font-size: 12px; font-weight: bold;">Topic:</span>
            <span style="font-size: 12px">@topic</span>
            <span style="font-size: 12px; font-weight: bold;">Title:</span>
            <span style="font-size: 12px">@title</span>
            <span style="font-size: 12px; font-weight: bold;">Year:</span>
            <span style="font-size: 12px">@year</span>
        </div>
    </div>
    """)
tools_emb = [hover_emb, 'pan', 'wheel_zoom', 'reset']
plot_emb = figure(plot_width=700, plot_height=700, tools=tools_emb, title='Papers')
plot_emb.circle('x', 'y', size='size', fill_color='colors', 
                 alpha='alpha', line_alpha=0, line_width=0.01, source=source, name="df", legend='topic')

for i in range(n_topics):
    plot_emb.cross(x=centroids[i,0], y=centroids[i,1], size=15, color='black', line_width=2, angle=0.79)
plot_emb.legend.location = "bottom_left"
plot_emb.legend.label_text_font_size= "8pt"
plot_emb.legend.spacing = -5
plot_emb.x_range = Range1d(-9, 7)
plot_emb.y_range = Range1d(-9, 7)

callback = CustomJS(args=dict(source=source), code=
    """
    var data = source.data;
    var f = cb_obj.value
    x = data['x']
    y = data['y']
    colors = data['colors']
    alpha = data['alpha']
    title = data['title']
    year = data['year']
    size = data['size']
    for (i = 0; i < x.length; i++) {
        if (year[i] <= f) {
            alpha[i] = 0.9
            size[i] = 7
        } else {
            alpha[i] = 0.05
            size[i] = 4
        }
    }
    source.change.emit();
    """)

slider = Slider(start=df.year.min()-1, end=df.year.max(), value=2016, step=1, title="Before year")
slider.js_on_change('value', callback)

layout = column(slider, plot_emb)
show(layout)


# #### 5.2.3. Static Picture

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
get_ipython().run_line_magic('matplotlib', 'inline')

legend_list = []
for color in all_palettes['Category20'][20][:n_topics]:   
    legend_list.append(mpatches.Ellipse((0, 0), 1, 1, fc=color))
    
fig,ax = plt.subplots(figsize=(12,13))
ax.scatter(embedding.x, embedding.y, c=my_colors, alpha=0.7)
ax.scatter(centroids[:,0], centroids[:,1], c='black', s=100, alpha=0.7, marker='x')
ax.set_title('11 topics found via NMF');
fig.legend(legend_list, topics, loc=(0.09,0.87), ncol=4)
plt.subplots_adjust(top=0.82)
plt.suptitle("NIPS clustered by topic", **{'fontsize':'14','weight':'bold'});
plt.figtext(.51,0.95, 'topic modeling with NMF + 2D-embedding with UMAP', 
            **{'fontsize':'12','weight':'light'}, ha='center');


# ### 5.3. NMF-23

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.decomposition import NMF\nn_topics=23\nn_top_words = 15\nnmf = NMF(n_components=n_topics, random_state=42, alpha=.1, l1_ratio=.5).fit(tfidf)\nnmf_embedding = nmf.transform(tfidf)\nfeature_names = tfidf_vectorizer.get_feature_names()\nprint("Topics found via NMF:")\nfor topic_idx, topic in enumerate(nmf.components_):\n    print("\\nTopic {}:".format(topic_idx))\n    print(" ".join([\'[{}]\'.format(feature_names[i]) for i in topic.argsort()[:-n_top_words - 1:-1]]))\nprint()')


# In[ ]:


topics = ['Optimization Algorithms',
          'Neural Networks',
          'Reinforcement Learning',
          'Image Recognition', 
          'Probabilistic Methods',
          'Visual Neurons',
          'Graph/Networks',
          'Classification Problems',          
          'Kernel Methods',
          'Bayesian Methods',
          'Multiiarm Bandits',
          'General Neurons',          
          'Clastering Methods',
          'Matrix Decompositions',
          'Control Theory',
          'Topic Modeling',          
          'Tree Methods',
          'Greedy Algorithms',
          'Speech Recognition',
          'Dimensionality Reduction',          
          'Chips/Circuit',
          'Game Theory',
          'Feature Engineering'
         ]


# #### 5.3.2. Bokeh interactive plot

# In[ ]:


centroids = umap_embr.transform(nmf.components_)
embedding['hue'] = nmf_embedding.argmax(axis=1)
my_colors = [(all_palettes['Category20'][20] + all_palettes['Category20'][20])[i] for i in embedding.hue]
source = ColumnDataSource(
        data=dict(
            x = embedding.x,
            y = embedding.y,
            colors = my_colors,
            topic = [topics[i] for i in embedding.hue],
            title = df.title,
            year = df.year,
            alpha = [0.7] * embedding.shape[0],
            size = [7] * embedding.shape[0]
        )
    )
hover_emb = HoverTool(names=["df"], tooltips="""
    <div style="margin: 10">
        <div style="margin: 0 auto; width:300px;">
            <span style="font-size: 12px; font-weight: bold;">Topic:</span>
            <span style="font-size: 12px">@topic</span>
            <span style="font-size: 12px; font-weight: bold;">Title:</span>
            <span style="font-size: 12px">@title</span>
            <span style="font-size: 12px; font-weight: bold;">Year:</span>
            <span style="font-size: 12px">@year</span>
        </div>
    </div>
    """)
tools_emb = [hover_emb, 'pan', 'wheel_zoom', 'reset']
plot_emb = figure(plot_width=700, plot_height=700, tools=tools_emb, title='Papers')
plot_emb.circle('x', 'y', size='size', fill_color='colors', 
                 alpha='alpha', line_alpha=0, line_width=0.01, source=source, name="df", legend='topic')

for i in range(n_topics):
    plot_emb.cross(x=centroids[i,0], y=centroids[i,1], size=15, color='black', line_width=2, angle=0.79)
plot_emb.legend.location = "bottom_left"
plot_emb.legend.label_text_font_size= "8pt"
plot_emb.legend.spacing = -5
plot_emb.x_range = Range1d(-9, 7)
plot_emb.y_range = Range1d(-9, 7)

callback = CustomJS(args=dict(source=source), code=
    """
    var data = source.data;
    var f = cb_obj.value
    x = data['x']
    y = data['y']
    colors = data['colors']
    alpha = data['alpha']
    title = data['title']
    year = data['year']
    size = data['size']
    for (i = 0; i < x.length; i++) {
        if (year[i] <= f) {
            alpha[i] = 0.9
            size[i] = 7
        } else {
            alpha[i] = 0.05
            size[i] = 4
        }
    }
    source.change.emit();
    """)

slider = Slider(start=df.year.min()-1, end=df.year.max(), value=2016, step=1, title="Before year")
slider.js_on_change('value', callback)

layout = column(slider, plot_emb)
show(layout)


# #### 5.3.3. Static Picture

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
get_ipython().run_line_magic('matplotlib', 'inline')

legend_list = []
for color in (all_palettes['Category20'][20] + all_palettes['Category20'][20])[:n_topics]:   
    legend_list.append(mpatches.Ellipse((0, 0), 1, 1, fc=color))
    
fig,ax = plt.subplots(figsize=(12,13))
ax.scatter(embedding.x, embedding.y, c=my_colors, alpha=0.7)
ax.scatter(centroids[:,0], centroids[:,1], c='black', s=100, alpha=0.7, marker='x')
ax.set_title('23 topics found via NMF');
fig.legend(legend_list, topics, loc=(0.075,0.835), ncol=4)
plt.subplots_adjust(top=0.82)
plt.suptitle("NIPS clustered by topic", **{'fontsize':'14','weight':'bold'});
plt.figtext(.51,0.95, 'topic modeling with NMF + 2D-embedding with UMAP', 
            **{'fontsize':'12','weight':'light'}, ha='center');

