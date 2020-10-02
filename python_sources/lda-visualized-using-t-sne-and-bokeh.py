#!/usr/bin/env python
# coding: utf-8

# This kernel was inspired by [NIPS papers visualized with NMF and t-SNE
# ](https://www.kaggle.com/dschniertshauer/nips-papers-visualized-with-nmf-and-t-sne) (Lurchi).

# In[ ]:


import pandas as pd
import numpy as np
# LDA, tSNE
from sklearn.manifold import TSNE
from gensim.models.ldamodel import LdaModel
# NLTK
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import re
# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
# Bokeh
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, CustomJS, ColumnDataSource, Slider
from bokeh.layouts import column
from bokeh.palettes import all_palettes
output_notebook()


# ## Loading data
# Let's load the dataset with papers and glimpse some first rows of a paper.

# In[ ]:


df = pd.read_csv("../input/papers.csv")
print(df.paper_text[0][:500])


# ## Processing
# Here we'll process our corpus using some standard technics ...

# ### Initial cleaning
# Just removing numbers and reducing all words to the lowercase. Let also see what we'll get:

# In[ ]:


get_ipython().run_cell_magic('time', '', "# Removing numerals:\ndf['paper_text_tokens'] = df.paper_text.map(lambda x: re.sub(r'\\d+', '', x))\n# Lower case:\ndf['paper_text_tokens'] = df.paper_text_tokens.map(lambda x: x.lower())\nprint(df['paper_text_tokens'][0][:500])")


# ### Tokenize
# Spliting texts into separete words, also removing punctuanions and other stuff. After that procedure we should obtain texts as lists of words in lowercase:

# In[ ]:


get_ipython().run_cell_magic('time', '', "df['paper_text_tokens'] = df.paper_text_tokens.map(lambda x: RegexpTokenizer(r'\\w+').tokenize(x))\nprint(df['paper_text_tokens'][0][:25])")


# ### Stemming
# Stemming is the process of reducing inflected (or sometimes derived) words to their word stem, base or root form ... The stem need not be identical to the morphological root of the word (see  [[Wikipedia]](https://en.wikipedia.org/wiki/Stemming) for more details). We'll use `SnowballStemmer` from `nltk` package.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'snowball = SnowballStemmer("english")  \ndf[\'paper_text_tokens\'] = df.paper_text_tokens.map(lambda x: [snowball.stem(token) for token in x])\nprint(df[\'paper_text_tokens\'][0][:25])')


# ### Stop words
#  Removing common English words like  `and`, `the`, `of` and so on.

# In[ ]:


get_ipython().run_cell_magic('time', '', "stop_en = stopwords.words('english')\ndf['paper_text_tokens'] = df.paper_text_tokens.map(lambda x: [t for t in x if t not in stop_en]) \nprint(df['paper_text_tokens'][0][:25])")


# ### Final cleaning
# Here we'll remove all "extremely short" words (that have less than 2 characters):

# In[ ]:


get_ipython().run_cell_magic('time', '', "df['paper_text_tokens'] = df.paper_text_tokens.map(lambda x: [t for t in x if len(t) > 1])\nprint(df['paper_text_tokens'][0][:25])")


# ## LDA
# Finally, let's use LDA ([Latent Dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)) to extract topic structure from the corpus of texts.

# In[ ]:


from gensim import corpora, models
np.random.seed(2017)
texts = df['paper_text_tokens'].values
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
ldamodel = models.ldamodel.LdaModel(corpus, id2word=dictionary, 
                                    num_topics=8, passes=5, minimum_probability=0)


# In[ ]:


ldamodel.print_topics()


# Refactoring results of LDA into numpy matrix (`number_of_papers` x `number_of_topics`).

# In[ ]:


hm = np.array([[y for (x,y) in ldamodel[corpus[i]]] for i in range(len(corpus))])


# And reduce dimensionality using t-SNE algorithm:

# In[ ]:


tsne = TSNE(random_state=2017, perplexity=30, early_exaggeration=120)
embedding = tsne.fit_transform(hm)
embedding = pd.DataFrame(embedding, columns=['x','y'])
embedding['hue'] = hm.argmax(axis=1)


# ## Ploting
# Using Bokeh for scatter plot with interactions. Hover mouse over a dot to see the title of the respective paper:

# In[ ]:


source = ColumnDataSource(
        data=dict(
            x = embedding.x,
            y = embedding.y,
            colors = [all_palettes['Set1'][8][i] for i in embedding.hue],
            title = df.title,
            year = df.year,
            alpha = [0.9] * embedding.shape[0],
            size = [7] * embedding.shape[0]
        )
    )
hover_tsne = HoverTool(names=["df"], tooltips="""
    <div style="margin: 10">
        <div style="margin: 0 auto; width:300px;">
            <span style="font-size: 12px; font-weight: bold;">Title:</span>
            <span style="font-size: 12px">@title</span>
            <span style="font-size: 12px; font-weight: bold;">Year:</span>
            <span style="font-size: 12px">@year</span>
        </div>
    </div>
    """)
tools_tsne = [hover_tsne, 'pan', 'wheel_zoom', 'reset']
plot_tsne = figure(plot_width=700, plot_height=700, tools=tools_tsne, title='Papers')
plot_tsne.circle('x', 'y', size='size', fill_color='colors', 
                 alpha='alpha', line_alpha=0, line_width=0.01, source=source, name="df")

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

slider = Slider(start=df.year.min(), end=df.year.max(), value=2016, step=1, title="Before year")
slider.js_on_change('value', callback)

layout = column(slider, plot_tsne)
show(layout)


# In[ ]:




