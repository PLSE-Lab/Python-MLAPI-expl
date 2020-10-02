#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import gc
import re
import folium
from scipy import stats

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import scipy as sp
import pandas as pd

import pycountry
from sklearn import metrics
from sklearn.utils import shuffle
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import nltk
from textblob import TextBlob
from wordcloud import WordCloud
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import random
import networkx as nx
from pandas import Timestamp

import requests
from IPython.display import HTML


# In[ ]:


import seaborn as sns
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

tqdm.pandas()
np.random.seed(0)
get_ipython().run_line_magic('env', 'PYTHONHASHSEED=0')

import warnings
warnings.filterwarnings("ignore")


# ### Load data

# In[ ]:


DATA_PATH = "../input/CORD-19-research-challenge/"
CLEAN_DATA_PATH = "../input/cord-19-eda-parse-json-and-generate-clean-csv/"

pmc_df = pd.read_csv(CLEAN_DATA_PATH + "clean_pmc.csv")
biorxiv_df = pd.read_csv(CLEAN_DATA_PATH + "biorxiv_clean.csv")
comm_use_df = pd.read_csv(CLEAN_DATA_PATH + "clean_comm_use.csv")
noncomm_use_df = pd.read_csv(CLEAN_DATA_PATH + "clean_noncomm_use.csv")

papers_df = pd.concat([pmc_df,
                       biorxiv_df,
                       comm_use_df,
                       noncomm_use_df], axis=0).reset_index(drop=True)


# ## Abstracts <a id="1.3"></a>
# 
# Every research paper has an abstract at the start, which briefly summarizes the contents and ideas presented in the paper. These abstracts can be a great source of insights and solutions (as we will see later). First, I will do some basic visualization of the abstracts in the dataset.

# ### Abstract words distribution

# In[ ]:


def new_len(x):
    if type(x) is str:
        return len(x.split())
    else:
        return 0

papers_df["abstract_words"] = papers_df["abstract"].apply(new_len)
nums = papers_df.query("abstract_words != 0 and abstract_words < 500")["abstract_words"]
fig = ff.create_distplot(hist_data=[nums],
                         group_labels=["All abstracts"],
                         colors=["coral"])

fig.update_layout(title_text="Abstract words", xaxis_title="Abstract words", template="simple_white", showlegend=False)
fig.show()


# In the above distribution plot, we can see that the abstract length has a roughly normal distribution with several minor peaks on either side of the mean. The probability density peaks at around 200 words, indicating that this is the most plausible value.

# In[ ]:


biorxiv_df["abstract_words"] = biorxiv_df["abstract"].apply(new_len)
nums_1 = biorxiv_df.query("abstract_words != 0 and abstract_words < 500")["abstract_words"]
pmc_df["abstract_words"] = pmc_df["abstract"].apply(new_len)
nums_2 = pmc_df.query("abstract_words != 0 and abstract_words < 500")["abstract_words"]
comm_use_df["abstract_words"] = comm_use_df["abstract"].apply(new_len)
nums_3 = comm_use_df.query("abstract_words != 0 and abstract_words < 500")["abstract_words"]
noncomm_use_df["abstract_words"] = noncomm_use_df["abstract"].apply(new_len)
nums_4 = noncomm_use_df.query("abstract_words != 0 and abstract_words < 500")["abstract_words"]
fig = ff.create_distplot(hist_data=[nums_1, nums_2, nums_3, nums_4],
                         group_labels=["Biorxiv", "PMC", "Commerical", "Non-commercial"],
                         colors=px.colors.qualitative.Plotly[4:], show_hist=False)

fig.update_layout(title_text="Abstract words vs. Paper type", xaxis_title="Abstract words", template="plotly_white")
fig.show()


# This plot shows the abstract length distribution for different research paper types (BiorXiv, PMC, Commercial, and Non-commercial). The abstract of commerical papers seem to longest on average, followed by non-commercial, BiorXiv, and PMC (in descending order).

# # Finding cures for COVID-19 <a id="2"></a>
# 
# Now, I will leverage the power of unsupervised machine learning to try and find possible cures (medicines and drugs) to COVID-19.

# ## Unsupervised NLP and Word2Vec <a id="2.1"></a>
# 
# Unsupervised NLP involves the analysis of unlabeled language data. Certain techniques can be used to derive insights from a large corpus of text. One such method is called **Word2Vec**. Word2Vec is a neural network architecture trained on thousands of sentences of text. After training, the neural network finds the **optimal vector representation** of each word in the corpus. These vectors are meant to reflect the meaning of the word. Words with similar meanings have similar vectors. 

# <center><img src="https://i.imgur.com/sZP4N8S.png" width="800px"></center>

# As I stated earlier, each word is associated with a vector. Amazingly, these vectors can also encode relationships and analogies between words. The diagram below iillustrates some examples of linear vector relationships representing the relationships between words.

# <center><img src="https://i.imgur.com/JHCOaan.png" width="800px"></center>

# In the above image, we can see that word vectors can reflect relationships such as "King is to Queen as Man is to Woman" or "Italy is to Rome" as "Germany is to Berlin". These vectors can be also be used to find unknown relationships between words. These unknown relationships may help us find latent knowledge in research papers and find drugs that can possibly cure COVID_19!

# ## Using Word2Vec to find cures <a id="2.2"></a>
# 
# We can take advantage of these intricate relationships between word vectors to find cures for COVID-19. The steps are as follows:
# 
# 1. Find common related to the study of COVID-19, such as "infection", "CoV", "viral", etc.
# 2. Find the words with lowest Euclidean distance to these words (most similar words).
# 3. Finally, find the words most similar to these words (second order similarity). These words will hopefully contain potential COVID-19 cures.
# 
# Note that the similarity between two Word2Vec vectors is calculated using the formula below (where *u* and *v* are the word vectors).
# 
# <center><img src="https://i.imgur.com/wBuMMS9.png" width="450px"></center>

# The entire process can be summarized with the flowchart below. (the same steps as given above)
# 
# <center><img src="https://i.imgur.com/l8b6enq.png" width="450px"></center>

# The approach detailed above is actually inspired by a research paper called ["Unsupervised word embeddings capture latent knowledge from materials science literature"](https://www.nature.com/articles/s41586-019-1335-8), where the authors find new materials with desirable properties (such as thermoelectricity) solely based on a large corpus materials science literature. These materials were never used for these purposes before, but they outperform old materials by a large margin. I hope to emulate the same method to look for COVID-19 cures. The diagram below illustrates what the authors did in their research.
# 
# <center><img src="https://i.imgur.com/TjXOhuJ.png" width="400px"></center>

# In the diagram above, we can see that the authors found two levels of words similar to "thermoelectric" in a heirarchical manner. The second order similar words contained compounds like Li<sub>2</sub>CuSb, Cu<sub>7</sub>Te<sub>5</sub>, and CsAgGa<sub>2</sub>Se<sub>4</sub>, which turned out to be very good thermoelectric materials in real life.

# ### Word cloud of abstracts

# In[ ]:


def nonan(x):
    if type(x) == str:
        return x.replace("\n", "")
    else:
        return ""

text = ' '.join([nonan(abstract) for abstract in papers_df["abstract"]])
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text)
fig = px.imshow(wordcloud)
fig.update_layout(title_text='Common words in abstracts')


# First, we need to find the most common words in the corpus to continue our analysis. From the word cloud above, we can see that "infection", "cell", "virus", and "protein" are among the most common words in COVID-19 research paper abstracts. These words will form our "keyword" list.

# In[ ]:


def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ

    elif nltk_tag.startswith('V'):
        return wordnet.VERB

    elif nltk_tag.startswith('N'):
        return wordnet.NOUN

    elif nltk_tag.startswith('R'):
        return wordnet.ADV

    else:          
        return None

def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []

    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))

    return " ".join(lemmatized_sentence)

def clean_text(abstract):
    abstract = abstract.replace(". ", " ").replace(", ", " ").replace("! ", " ")                       .replace("? ", " ").replace(": ", " ").replace("; ", " ")                       .replace("( ", " ").replace(") ", " ").replace("| ", " ").replace("/ ", " ")
    if "." in abstract or "," in abstract or "!" in abstract or "?" in abstract or ":" in abstract or ";" in abstract or "(" in abstract or ")" in abstract or "|" in abstract or "/" in abstract:
        abstract = abstract.replace(".", " ").replace(",", " ").replace("!", " ")                           .replace("?", " ").replace(":", " ").replace(";", " ")                           .replace("(", " ").replace(")", " ").replace("|", " ").replace("/", " ")
    abstract = abstract.replace("  ", " ")
    
    for word in list(set(stopwords.words("english"))):
        abstract = abstract.replace(" " + word + " ", " ")

    return lemmatize_sentence(abstract).lower()

def get_similar_words(word, num):
    vec = model_wv_df[word].T
    distances = np.linalg.norm(model_wv_df.subtract(model_wv_df[word], 
                                                    axis=0).values, axis=0)

    indices = np.argsort(distances)
    top_distances = distances[indices[1:num+1]]
    top_words = model_wv_vocab[indices[1:num+1]]
    return top_words

def visualize_word_list(color, word):
    top_words = get_similar_words(word, num=6)
    relevant_words = [get_similar_words(word, num=8) for word in top_words]
    fig = make_subplots(rows=3, cols=2, subplot_titles=tuple(top_words), vertical_spacing=0.05)
    for idx, word_list in enumerate(relevant_words):
        words = [word for word in word_list if word in model_wv_vocab]
        X = model_wv_df[words].T
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)
        df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
        df["Word"] = word_list
        word_emb = df[["Component 1", "Component 2"]].loc[0]
        df["Distance"] = np.sqrt((df["Component 1"] - word_emb[0])**2 + (df["Component 2"] - word_emb[1])**2)
        plot = px.scatter(df, x="Component 1", y="Component 2", text="Word", color="Distance", color_continuous_scale=color, size="Distance")
        plot.layout.title = top_words[idx]
        plot.update_traces(textposition='top center')
        plot.layout.xaxis.autorange = True
        plot.data[0].marker.line.width = 1
        plot.data[0].marker.line.color = 'rgb(0, 0, 0)'
        fig.add_trace(plot.data[0], row=(idx//2)+1, col=(idx%2)+1)
    fig.layout.coloraxis.showscale = False
    fig.update_layout(height=1400, title_text="2D PCA of words related to {}".format(word), paper_bgcolor="#f0f0f0", template="plotly_white")
    return fig

def visualize_word(color, word):
    top_words = get_similar_words(word, num=20)
    words = [word for word in top_words if word in model_wv_vocab]
    X = model_wv_df[words].T
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
    df["Word"] = top_words
    if word == "antimalarial":
        df = df.query("Word != 'anti-malarial' and Word != 'anthelmintic'")
    if word == "doxorubicin":
        df = df.query("Word != 'anti-rotavirus'")
    word_emb = df[["Component 1", "Component 2"]].loc[0]
    df["Distance"] = np.sqrt((df["Component 1"] - word_emb[0])**2 + (df["Component 2"] - word_emb[1])**2)
    fig = px.scatter(df, x="Component 1", y="Component 2", text="Word", color="Distance", color_continuous_scale=color, size="Distance")
    fig.layout.title = word
    fig.update_traces(textposition='top center')
    fig.layout.xaxis.autorange = True
    fig.layout.coloraxis.showscale = True
    fig.data[0].marker.line.width = 1
    fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
    fig.update_layout(height=800, title_text="2D PCA of words related to {}".format(word), template="plotly_white", paper_bgcolor="#f0f0f0")
    fig.show()


# ### Load pretrained Word2Vec model (200D vectors) / Train on Corpus

# In[ ]:


#lemmatizer = WordNetLemmatizer()

#def get_words(abstract):
#    return clean_text(nonan(abstract)).split(" ")

#words = papers_df["abstract"].progress_apply(get_words)
#model = Word2Vec(words, size=200, sg=1, min_count=1, window=8, hs=0, negative=15, workers=1)

model = Word2Vec.load("../input/covid19-challenge-trained-w2v-model/covid.w2v")
#model_wv_vocab = pd.read_csv("../input/word2vec-results-1/vocab.csv").values[:, 0]
#model_wv_df = pd.DataFrame(np.transpose(model_wv), columns=model_wv_vocab)


# ### Visualize most similar words to keywords

# In[ ]:


keywords = ["infection", "cell", "protein", "virus",            "disease", "respiratory", "influenza", "viral",            "rna", "patient", "pathogen", "human", "medicine",            "cov", "antiviral"]

print("Most similar words to keywords")
print("")

top_words_list = []
for word in keywords:
    print(word, model.wv.most_similar(word, topn=5))


# These words will form the next batch of words, which we will analyze to find cures to COVID-19.

# ### PCA
# 
# PCA is a dimensionality reduction method which takes vectors with several dimensions and compresses it into a smaller vector (with 2 or 3 dimensions) while preserving most of the information in the original vector (using some linear algebra). PCA makes visualization easier while dealing with high-dimensional data, such as Word2Vec vectors.
# 
# <center><img src="https://i.imgur.com/CKWFUyd.png" width="400px"></center>

# ### 2D PCA of keyword vectors

# In[ ]:


words = [word for word in keywords if word in model_wv_vocab]
X = model_wv_df[words].T
pca = PCA(n_components=2)
result = pca.fit_transform(X)
df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
df["Word"] = keywords
df["Distance"] = np.sqrt(df["Component 1"]**2 + df["Component 2"]**2)
fig = px.scatter(df, x="Component 1", y="Component 2", text="Word", color="Distance", color_continuous_scale="agsunset",size="Distance")
fig.update_traces(textposition='top center')
fig.layout.xaxis.autorange = True
fig.data[0].marker.line.width = 1
fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
fig.update_layout(height=800, title_text="2D PCA of Word2Vec embeddings", template="plotly_white", paper_bgcolor="#f0f0f0")
fig.show()


# In the above plot, we can see the 2D PCA of the keywords' vectors.
# 
# 1. The words "virus", "viral", and "CoV" form a cluster in the bottom-right part of the plot, indicating that they have similar meanings. This makes sense because CoV is a virus.
# 2. The words "medicine" and "patient" are both on the far left end of the image because these words are used together very frequently.
# 3. The "pathogen", "influenza", and "respiratory" form a cluster in the bottom-left part of the plot, indicating that they have similar meanings. This makes sense because influenza is a repsiratory disease.
# 
# These abstract linguistic relationships are successfully represented by word vectors.

# ### 3D PCA of keyword vectors

# In[ ]:


words = [word for word in keywords if word in model_wv_vocab]
X = model_wv_df[words].T
pca = PCA(n_components=3)
result = pca.fit_transform(X)
df = pd.DataFrame(result, columns=["Component 1", "Component 2", "Component 3"])
df["Word"] = keywords
df["Distance"] = np.sqrt(df["Component 1"]**2 + df["Component 2"]**2 + df["Component 3"]**2)
fig = px.scatter_3d(df, x="Component 1", y="Component 2", z="Component 3", text="Word", color="Distance", color_continuous_scale="agsunset")
fig.update_traces(textposition='top left')
fig.layout.coloraxis.showscale = False
fig.layout.xaxis.autorange = True
fig.update_layout(height=800, title_text="3D PCA of Word2Vec embeddings", template="plotly")
fig.show()


# I have plotted the 3D PCA above. The clustering seems to be very similar to that in 2D PCA. More dimensions usually ensure better clustering and word representation, but it comes at the cost of higher dimensionality and less intuitive visualization.

# ### 2D PCA of words related to keywords
# 
# Now, I will pick up a few keywords and analyze the PCA of words similar to them, making conclusions and inferences as I go.

# ### 2D PCA of words similar to influenza

# In[ ]:


words = [word for word in top_words_list[6] if word in model_wv_vocab]
X = model_wv_df[words].T
pca = PCA(n_components=2)
result = pca.fit_transform(X)
df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
df["Word"] = top_words_list[6]
word_emb = df[["Component 1", "Component 2"]].loc[0]
df["Distance"] = np.sqrt((df["Component 1"] - word_emb[0])**2 + (df["Component 2"] - word_emb[1])**2)
fig = px.scatter(df.query("Word != 'uenza'"), x="Component 1", y="Component 2", text="Word", color="Distance", color_continuous_scale="aggrnyl",size="Distance")

"""for row in range(len(df)):
    fig.add_shape(
                type="line",
                x0=word_emb[0],
                y0=word_emb[1],
                x1=df["Component 1"][row],
                y1=df["Component 2"][row],
                line=dict(
                    color="Green",
                    width=0.75,
                    dash="dot"
                )
    )"""

fig.update_traces(textposition='top center')
fig.layout.xaxis.autorange = True
fig.data[0].marker.line.width = 1
fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
fig.update_layout(height=800, title_text="2D PCA of words related to {}".format(keywords[6]), template="plotly_white", paper_bgcolor="#f0f0f0")
fig.show()


# I have plotted the 2D PCA of the words most similar to influenza above.
# 
# 1. The words "H2N2", "PDM", "PDM2009", "H7N7", and "swine origin" form a very dense cluster in the bottom-left corner of the plot. This makes sense because H2N2 and H7N7 are both subtypes of Influenza and they have their origin in swines. Note that "PDM" stands for pandemic.
# 2. The remaining words are very far away from this cluster. For example, the word "flu" is far away from this cluster because it is a general term which is not equivalent to any specific type of flu or influenza.

# ### 2D PCA of words similar to RNA

# In[ ]:


words = [word for word in top_words_list[8] if word in model_wv_vocab]
X = model_wv_df[words].T
pca = PCA(n_components=2)
result = pca.fit_transform(X)
df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
df["Word"] = top_words_list[8]
word_emb = df[["Component 1", "Component 2"]].loc[0]
df["Distance"] = np.sqrt((df["Component 1"] - word_emb[0])**2 + (df["Component 2"] - word_emb[1])**2)
fig = px.scatter(df[1:].query("Word != 'abstractrna'"), x="Component 1", y="Component 2", text="Word", color="Distance", color_continuous_scale="agsunset",size="Distance")

"""for row in range(len(df)):
    fig.add_shape(
                type="line",
                x0=word_emb[0],
                y0=word_emb[1],
                x1=df["Component 1"][row],
                y1=df["Component 2"][row],
                line=dict(
                    color="MediumPurple",
                    width=0.75,
                    dash="dot"
                )
    )"""

fig.update_traces(textposition='top center')
fig.layout.xaxis.autorange = True
fig.data[0].marker.line.width = 1
fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
fig.update_layout(height=800, title_text="2D PCA of words related to {}".format(keywords[8]), template="plotly_white", paper_bgcolor="#f0f0f0")
fig.show()


# I have plotted the 2D PCA of the words most similar to RNA above. We cannot see an clear clustering in the plot above, but we can see that few words similar to RNA appear in the graph. For example, the words "ssRNA" (single-stranded RNA) and "vRNA" (viral RNA), which are types of RNA (ribonucleic acid). We also see words like "negative-strand" and "negative-sense", When we put all these terms together, it makes sense because they are deeply related. The genome of the influenza virus is in fact composed of eight negative-strand vRNA!

# ### 2D PCA of words similar to CoV

# In[ ]:


words = [word for word in top_words_list[-2] if word in model_wv_vocab]
X = model_wv_df[words].T
pca = PCA(n_components=2)
result = pca.fit_transform(X)
df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
df["Word"] = top_words_list[-2]
word_emb = df[["Component 1", "Component 2"]].loc[0]
df["Distance"] = np.sqrt((df["Component 1"] - word_emb[0])**2 + (df["Component 2"] - word_emb[1])**2)
fig = px.scatter(df[1:], x="Component 1", y="Component 2", text="Word", color="Distance", color_continuous_scale="oryel",size="Distance")


"""for row in range(len(df)):
    fig.add_shape(
                type="line",
                x0=word_emb[0],
                y0=word_emb[1],
                x1=df["Component 1"][row],
                y1=df["Component 2"][row],
                line=dict(
                    color="Orange",
                    width=0.75,
                    dash="dot"
                )
    )"""

fig.update_traces(textposition='top center')
fig.layout.xaxis.autorange = True
fig.data[0].marker.line.width = 1
fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
fig.update_layout(height=800, title_text="2D PCA of words related to {}".format(keywords[-2]), template="plotly_white", paper_bgcolor="#f0f0f0")
fig.show()


# I have plotted the 2D PCA of the words most similar to CoV (stands for **CO**rona**V**irus) above.
# 
# 1. We can see few words like "coronavirus", "SARS-CoV", and "coronaviral" which are almost synonymal with CoV. These words are surprisingly very close to "CoV" in the vector space.
# 2. We can also see a clear cluster in the bottom-left corner of the plot, and these words are also closely linked with the word "CoV".

# ### 2D PCA of words related to virus

# In[ ]:


words = [word for word in top_words_list[3] if word in model_wv_vocab]
X = model_wv_df[words].T
pca = PCA(n_components=2)
result = pca.fit_transform(X)
df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
df["Word"] = top_words_list[3]
word_emb = df[["Component 1", "Component 2"]].loc[0]
df["Distance"] = np.sqrt((df["Component 1"] - word_emb[0])**2 + (df["Component 2"] - word_emb[1])**2)
fig = px.scatter(df[1:], x="Component 1", y="Component 2", text="Word", color="Distance", color_continuous_scale="bluered",size="Distance")

"""for row in range(len(df)):
    fig.add_shape(
                type="line",
                x0=word_emb[0],
                y0=word_emb[1],
                x1=df["Component 1"][row],
                y1=df["Component 2"][row],
                line=dict(
                    color="Purple",
                    width=0.75,
                    dash="dot"
                )
    )"""

fig.update_traces(textposition='top center')
fig.layout.xaxis.autorange = True
fig.data[0].marker.line.width = 1
fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
fig.update_layout(height=800, title_text="2D PCA of words related to {}".format(keywords[3]), template="plotly_white", paper_bgcolor="#f0f0f0")
fig.show()


# I have plotted the 2D PCA of the words most similar to virus above. We cannot see any clear clustering, but we do see many types of viruses, such as "pneumovirus", "lyssavirus", "pox", "CPIV", and "HHV", appearing in the plot.
# 
# Now since we have visualized the PCA of words most similar to certain keywords, let us use the same strategy to find a possible medicine for COVID-19.

# ### 2D PCA of words related to antiviral

# In[ ]:


words = [word for word in top_words_list[-1] if word in model_wv_vocab]
X = model_wv_df[words].T
pca = PCA(n_components=2)
result = pca.fit_transform(X)
df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
df["Word"] = top_words_list[-1]
word_emb = df[["Component 1", "Component 2"]].loc[0]
df["Distance"] = np.sqrt((df["Component 1"] - word_emb[0])**2 + (df["Component 2"] - word_emb[1])**2)
fig = px.scatter(df[2:], x="Component 1", y="Component 2", text="Word", color="Distance", color_continuous_scale="viridis",size="Distance")

"""for row in range(len(df)):
    fig.add_shape(
                type="line",
                x0=word_emb[0],
                y0=word_emb[1],
                x1=df["Component 1"][row],
                y1=df["Component 2"][row],
                line=dict(
                    color="Purple",
                    width=0.75,
                    dash="dot"
                )
    )"""

fig.update_traces(textposition='top center')
fig.layout.xaxis.autorange = True
fig.data[0].marker.line.width = 1
fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
fig.update_layout(height=800, title_text="2D PCA of words related to {}".format(keywords[-1]), template="plotly_white", paper_bgcolor="#f0f0f0")
fig.show()


# I have plotted the 2D PCA of the words most similar to antiviral above. We can see a lot of different types of antivirals and other drugs in the plot, such as "saracatinib", an anti-malarial and anti-HIV drug. The list also includes "antiparasitic", "ant-HBV", and "anti-EV71".

# ### Second-order word similarities
# 
# Now, I will look at the words similar to the words found above (second order similarity) to hopefully, find potential cures for COVID-19.

# ### 2D PCA of words similar to words similar to antiviral

# In[ ]:


fig = visualize_word_list('agsunset', 'antiviral')
fig.update_layout(colorscale=dict(diverging=px.colors.diverging.Tealrose))


# We can see some amazing patterns in the plots above. We see certain drugs and chemicals that keep repeating, including "anti-malarial", "hydroxychloroquine", and "doxorubicin". It is amazing that these drugs have actually been successfully applied on COVID-19 patients across the world. There are cases of anti-malarial drugs working for COVID-19!
# 
# **The most common result above, "hydroxychloroquine", might just be the cure for COVID-19!**[](http://)

# ### 2D PCA of words similar to words similar to antimalarial

# In[ ]:


visualize_word('plotly3', 'antimalarial')


# In the plot above, we can see the words most similar to antimalarial. These are different drugs and medicines that are used to combat malaria, which may work for COVID-19, such as "amodiaquine", "hydroxychloroquine", and "nitazoxanide".

# # Takeaways <a id="4"></a>
# 
# 1. Several antimalarial drugs such as hydroxychloroquine might be potential drugs to cure COVID-19. Antimalarial drugs have been successfully tested on COVID-19 patients in certain countries.

# In[ ]:




