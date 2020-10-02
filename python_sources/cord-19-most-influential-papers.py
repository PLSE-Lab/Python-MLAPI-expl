#!/usr/bin/env python
# coding: utf-8

# # CORD-19 Most Influential Papers
# 
# ![CORD19](https://pages.semanticscholar.org/hs-fs/hubfs/covid-image.png?width=300&name=covid-image.png)
# 
# This notebook shows the most cited and influential papers within the CORD-19 dataset. This notebook focuses on citations related to articles tagged as COVID-19. The goal with this notebook is to give a good starting point in learning more about COVID-19. The more times researchers are citing a paper, the more likely it's to be good quality work that is worth reviewing. 
# 

# In[ ]:


# Install cord19q project
get_ipython().system('pip install git+https://github.com/neuml/cord19q')


# In[ ]:


from cord19q.highlights import Highlights
from cord19q.tokenizer import Tokenizer

from nltk.corpus import stopwords

from wordcloud import WordCloud

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pycountry

STOP_WORDS = set(stopwords.words("english")) 

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
countries = countries + ["USA"]

# Lookup country name for alpha code. If already an alpha code, return value
def countryname(x):
    country = pycountry.countries.get(alpha_3=x)
    return country.name if country else x
    
# Resolve alpha code for country name
def countrycode(x):
    return pycountry.countries.get(name=x).alpha_3

# Tokenize and filter only country names
def countrynames(x):
    x = x.lower()
    
    return [countryname(country) for country in countries if country.lower() in x]

# Word Cloud colors
def wcolors(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):
    colors = ["#7e57c2", "#03a9f4", "#011ffd", "#ff9800", "#ff2079"]
    return np.random.choice(colors)

# Word Cloud visualization
def wordcloud(df, title, recent):
    # Set random seed to have reproducible results
    np.random.seed(64)
    
    wc = WordCloud(
        background_color="white" if recent else "black",
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

# Map visualization
def mapplot(df, title, bartitle, color1, color2):
    fig = go.Figure(data=go.Choropleth(
        locations = df["Code"],
        z = df["Count"],
        text = df["Country"],
        colorscale = [(0, color1), (1, color2)],
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


# In[ ]:


# Build a word cloud for Top 25 cited articles
def citecloud(recent):
    # Connect to database
    db = sqlite3.connect("../input/cord-19-analysis-with-sentence-embeddings/cord19q/articles.sqlite")

    # Citations
    citations = pd.read_sql_query("select text from sections where article in " + 
                                "(select a.id from articles a join citations c on a.title = c.title " + 
                                 "where tags is %s null %s order by mentions desc limit 25)" % ("not" if recent else "", "and published <= '2020-01-01' and a.title != " + 
                                                                                                                        "'World Health Organization'" if not recent else ""), db)
    freqs = pd.Series(np.concatenate([tokenize(x) for x in citations.Text])).value_counts()
    wordcloud(freqs, "Most Frequent Words In Highly Cited %s Papers" % ("COVID-19" if recent else "Historical"), recent)


# In[ ]:


# Show top countries for Top 25 cited articles
def citemap(recent):
    # Connect to database
    db = sqlite3.connect("../input/cord-19-analysis-with-sentence-embeddings/cord19q/articles.sqlite")

    sections = pd.read_sql_query("select text from sections where article in (select id from articles a join citations c on a.title = c.title " + 
                                 "where tags is %s null %s order by mentions desc limit 25)" % ("not" if recent else "", "and published <= '2020-01-01' and a.title != " + 
                                                                                                                         "'World Health Organization'" if not recent else ""), db)
    
    # Filter tokens to only country names. Build dataframe of Country, Count, Code
    mentions = pd.Series(np.concatenate([countrynames(x) for x in sections.Text])).value_counts()
    mentions = mentions.rename_axis("Country").reset_index(name="Count")
    mentions["Code"] = [countrycode(x) for x in mentions["Country"]]

    mapplot(mentions, "Highly Cited %s Papers - Country Mentioned" % ("COVID-19" if recent else "Historical"), "Articles by Country", 
            "#fffde7" if recent else "#ffcdd2", "#f57f17" if recent else "#b71c1c")


# In[ ]:


import datetime
import os
import sqlite3

import pandas as pd

from IPython.core.display import display, HTML

# Workaround for mdv terminal width issue
os.environ["COLUMNS"] = "80"

from cord19q.query import Query

def design(df):
    # Study Design
    return "%s" % Query.design(df["Design"]) + ("<br/><br/>" + Query.text(df["Sample"]) if df["Sample"] else "")

def citations(recent):
    # Connect to database
    db = sqlite3.connect("../input/cord-19-analysis-with-sentence-embeddings/cord19q/articles.sqlite")

    # Citations
    citations = pd.read_sql_query("select published, authors, publication, a.title, reference, mentions as Cited from articles a join citations c on a.title = c.title " + 
                                  "where tags is %s null %s order by mentions desc limit 25" % ("not" if recent else "", "and published <= '2020-01-01' and a.title != " + 
                                                                                                                         "'World Health Organization'" if not recent else ""), db)

    citations["Published"] = citations["Published"].apply(Query.date)
    citations["Authors"] = citations["Authors"].apply(Query.authors)
    citations["Title"] = "<a href='" + citations["Reference"] + "'>" + citations["Title"] + "</a>"

    citations.style.bar(subset=["Cited"], color='#d65f5f')
    citations.style.hide_index()

    # Remove unnecessary columns
    citations = citations.drop("Reference", 1)

    # Set index to be 1-based
    citations.index = np.arange(1, len(citations) + 1)

    ## Show table as HTML
    display(HTML(citations.to_html(escape=False)))


# # How these rankings work
# Each paper in the CORD-19 dataset has a references section with citations. The citations for the full dataset have been loaded into a database via [another notebook](https://www.kaggle.com/davidmezzetti/cord-19-analysis-with-sentence-embeddings). Each paper that exists in the dataset is stored along with the number of times it's cited. This method doesn't count citations that are not in the CORD-19 dataset. It only considers a citation if it's a citation within a COVID-19 tagged paper. 
# 
# The rankings are broken into two sections, Highly Cited COVID-19 papers and Highly Cited historical papers within the CORD-19 corpus. 

# # Highly Cited COVID-19 Papers
# The following papers are the most cited recent papers. Papers that have been around longer will be cited more than recent papers. But even with that being said, these papers are typically well respected, full of good background information and educational on how we got here with COVID-19. Many of these papers discuss the origins of the virus when the world was just first finding out what was about to be unleashed. 

# ## Most Frequent Words in Highly Cited COVID-19 Papers
# The following wordcloud shows the most frequent words for the highly cited COVID-19 papers. Given that many of these articles are from when the outbreak first started, you'll see a lot of terms related to Wuhan and China. 

# In[ ]:


citecloud(True)


# ## Highly Cited COVID-19 Papers by Country Mentioned
# The following map shows the papers by country mentioned. Once again, China is mentioned significantly more given that the outbreak started there and the most cited papers cover the period of the initial outbreak.

# In[ ]:


citemap(True)


# ## The Highly Cited COVID-19 Papers
# Below is the table with the Top 25 papers. Based on the titles, you can once again see the theme of covering the initial outbreak. The cited column is the number of documents within the CORD-19 dataset that reference that document. 

# In[ ]:


citations(True)


# # Highly Cited Historical Papers
# In addition to the recent COVID-19 papers, the CORD-19 dataset has a number of historical articles on SARS, Coronaviruses and other related diseases. The following papers provide a good background on the general area of study. It's also informational on previous disease outbreaks (SARS, MERS, Ebola, Zika) and treatment methods tried during those outbreaks. You'll see some of the same drug names that are being discussed now as a possible treatment path.

# ## Most Frequent Words in Highly Cited Historical Papers
# The following wordcloud shows the most frequent words for the highly cited historical papers. The date range is much wider for this dataset but still mostly focused on coronaviruses. There is a lot of literature in this dataset related to the first outbreak of SARS and MERS. 

# In[ ]:


citecloud(False)


# ## Highly Cited Historical Papers by Country Mentioned
# The following map shows the papers by country mentioned. This time Saudi Arabia is a hotspot along with China. This is due to the locations of the first SARS outbreak in Hong Kong and the MERS outbreak in the Middle East. Notice there are also mentions of the US, Cuba and West Africa. This is due to the documents have text discussing the Zika outbreak and Ebola. 

# In[ ]:


citemap(False)


# ## The Highly Cited Historical Papers
# Below is the table with the Top 25 papers. If you've looked a lot at the COVID-19 dataset, you'll see a lot of the common themes, with even some of the same proposed drugs. Overall, a lot of good reading material on the general subject area. 

# In[ ]:


citations(False)

