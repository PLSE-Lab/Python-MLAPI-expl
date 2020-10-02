#!/usr/bin/env python
# coding: utf-8

# Sentence Embeddings Index with FastText + BM25: Should We Wear Masks?
# ======
# This notebook uses [David Mezzetti's cord19q text search engine](https://www.kaggle.com/davidmezzetti/cord-19-analysis-with-sentence-embeddings). 
# 
# ![CORD19](https://pages.semanticscholar.org/hs-fs/hubfs/covid-image.png?width=300&name=covid-image.png)
# 

# # Install from GitHub
# 
# ![](http://)Full source code for [cord19q](https://github.com/neuml/cord19q) is on GitHub and be installed into this notebook as follows:

# In[ ]:


# Install cord19q project
get_ipython().system('pip install git+https://github.com/neuml/cord19q')

# Install scispacy model
get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz')


# # Build SQLite articles database
# 
# The raw CORD-19 data is stored across a metadata.csv file and json files with the full text. This project uses [SQLite](https://www.sqlite.org/index.html) to aggregate and store the merged content.
# 
# The ETL process transforms the csv/json files into a SQLite database. The process iterates over each row in metadata.csv, extracts the column data and ensures it is not a pure duplicate (using the sha hash). This process will also load the full text if available. 

# In[ ]:


from cord19q.etl.execute import Execute as Etl

# Build SQLite database for metadata.csv and json full text files
Etl.run("../input/CORD-19-research-challenge", "cord19q")


# Upon completion, a database named articles.sqlite will be stored in the output directory under a sub-folder named cord19q.

# # Build Embedding Index
# 
# An embeddings index is created with [FastText](https://fasttext.cc/) + [BM25](https://en.wikipedia.org/wiki/Okapi_BM25). Background on this method can be found in this [Medium article](https://towardsdatascience.com/building-a-sentence-embedding-index-with-fasttext-and-bm25-f07e7148d240) and an existing repository using this method [codequestion](https://github.com/neuml/codequestion).
# 
# The embeddings index takes each COVID-19 tagged, non-labeled (not a question/fragment) section, tokenizes the text, and builds a sentence embedding. A sentence embedding is a BM25 weighted combination of the FastText vectors for each token in the sentence. The embeddings index takes the full corpus of these embeddings and builds a [Faiss](https://github.com/facebookresearch/faiss) index to enable similarity searching. 

# In[ ]:


import shutil

from cord19q.index import Index

# Copy vectors locally for predictable performance
shutil.copy("../input/cord19-fasttext-vectors/cord19-300d.magnitude", "/tmp")

# Build the embeddings index
Index.run("cord19q", "/tmp/cord19-300d.magnitude")


# In[ ]:


from cord19q.highlights import Highlights
from cord19q.tokenizer import Tokenizer

from wordcloud import WordCloud

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pycountry

# Tokenizes text and removes stopwords
def tokenize(text, case_sensitive=False):
    # Get list of accepted tokens
    tokens = [token for token in Tokenizer.tokenize(text) if token not in Highlights.STOP_WORDS]
    
    if case_sensitive:
        # Filter original tokens to preserve token casing
        return [token for token in text.split() if token.lower() in tokens]

    return tokens
    
# Country data
countries = [c.name for c in pycountry.countries]
countries = countries + [c.alpha_3 for c in pycountry.countries]

# Lookup country name for alpha code. If already an alpha code, return value
def countryname(x):
    country = pycountry.countries.get(alpha_3=x)
    return country.name if country else x
    
# Resolve alpha code for country name
def countrycode(x):
    return pycountry.countries.get(name=x).alpha_3

# Tokenize and filter only country names
def countrynames(x):
    return [countryname(token) for token in tokenize(x, True) if token in countries]

# Word Cloud colors
def wcolors(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):
    colors = ["#7e57c2", "#03a9f4", "#011ffd", "#ff9800", "#ff2079"]
    return np.random.choice(colors)

# Word Cloud visualization
def wordcloud(df, title = None):
    # Set random seed to have reproducible results
    np.random.seed(64)
    
    wc = WordCloud(
        background_color="white",
        max_words=200,
        max_font_size=40,
        scale=5,
        random_state=0
    ).generate_from_frequencies(df)

    wc.recolor(color_func=wcolors)
    
    fig = plt.figure(1, figsize=(15,15))
    plt.axis('off')

    if title:
        fig.suptitle(title, fontsize=14)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wc),
    plt.show()

# Dataframe plot
def plot(df, title, kind="bar", color="bbddf5"):
    # Remove top and right border
    ax = plt.axes()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set axis color
    ax.spines['left'].set_color("#bdbdbd")
    ax.spines['bottom'].set_color("#bdbdbd")

    df.plot(ax=ax, title=title, kind=kind, color=color);

# Pie plot
def pie(labels, sizes, title):
    patches, texts = plt.pie(sizes, colors=["#4caf50", "#ff9800", "#03a9f4", "#011ffd", "#ff2079", "#7e57c2", "#fdd835"], startangle=90)
    plt.legend(patches, labels, loc="best")
    plt.axis('equal')
    plt.tight_layout()
    plt.title(title)
    plt.show()
    
# Map visualization
def mapplot(df, title, bartitle):
    fig = go.Figure(data=go.Choropleth(
        locations = df["Code"],
        z = df["Count"],
        text = df["Country"],
        colorscale = [(0,"#fffde7"), (1,"#f57f17")],
        showscale = False,
        marker_line_color="darkgray",
        marker_line_width=0.5,
        colorbar_title = bartitle,
    ))

    fig.update_layout(
        title={
            'text': title,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        )
    )
    
    fig.show(config={"displayModeBar": False, "scrollZoom": False})


# ## Run a query
# Run a full query to ensure model is working.

# In[ ]:


from cord19q.query import Query

# Execute a test query
Query.run("healthy people should wear masks", 50, "cord19q")


# In[ ]:


Query.run("masks", 50, "cord19q")


# ## Conclusion
# - One advantage of universal use of face masks is that it prevents discrimination of individuals who wear masks when unwell because everybody is wearing a mask.
# - All patients and the people accompanying them are required to wear surgical masks.
# - Laboratory workers must wear the required personal protective equipment including long sleeved gowns, gloves, eye protection, and N95 masks.
# - Medical personnel working in this area should wear disposable work caps, N95 protective masks, isolation gowns, and disposable latex gloves, and strictly perform good hand
#  hygiene.
# - The World Health Organization (WHO) has recommended that healthcare staff wear appropriate eye protection, surgical masks, long-sleeved gowns and gloves when entering a room
#  containing suspected or confirmed COVID-19-infected individuals.
# - It is highly recommended that all healthcare providers wear N-95 mask, surgical cap, gown, eye goggles, shoe covers, double gloves, and PAPRs or full-face shield .
# - Basic PPE for the care of 1 patients with suspected or confirmed infection include N95 masks, protection goggles (or complete helmet), caps, gowns and gloves, and wearing two
#  gloves can be considered.
# - Since particles can penetrate even five surgical masks stacked together, health-care providers in direct contact with patients must wear N95 (series # 1860s) masks but not
#  surgical masks 118.
# - Staff coming into patient contact should, where appropriate, wear disposable personal protection including an isolation gown with fluid-resistant properties, over-gown gloves,
#  googles, and a fit-tested "filtering face piece" FFP3 mask.
# - In addition to masks, health-care providers should wear fitted isolation gowns in order to further reduce contact with viruses.
# 

# In[ ]:




