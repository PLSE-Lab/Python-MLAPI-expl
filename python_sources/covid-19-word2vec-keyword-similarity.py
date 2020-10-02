#!/usr/bin/env python
# coding: utf-8

# # COVID-19 papers Word2Vec model
# 
# Created by a TransUnion data scientist that believes that information can be used to change our world for the better. #InformationForGood
# 
# Model built from the [CORD-19 research challenge](kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) using [this](https://www.kaggle.com/elsonidoq/train-a-word2vec) notebook.
# 
# The approach has been adopted from [this](https://www.kaggle.com/tarunpaparaju/covid-19-dataset-gaining-actionable-insights) kernel.

# In[ ]:


from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")


# #### Loading trained Word2Vec model

# In[ ]:


model = Word2Vec.load('/kaggle/input/covid19-challenge-trained-w2v-model/covid.w2v')


# We'll use word similarity from the Word2Vec model

# In[ ]:


model.wv.most_similar('coronavirus', topn=10)


# In[ ]:


keywords = ["infection", "cell", "protein", "virus",            "disease", "respiratory", "influenza", "viral",            "rna", "patient", "pathogen", "human", "medicine",            "cov", "antiviral"]

print("Frequency of keyword & Most similar words")
print("")

#top_words_list = []
for word in keywords:
    top_words = model.wv.most_similar(word, topn=5)
    print(word + " - " + "frequency: ", model.wv.vocab[word].count)
    for idx, top_word in enumerate(top_words):
        print(str(idx+1) + ". " + top_word[0])
        #top_words_list.append(top_word[0])
    print("")


# ## 2-D PCA of keyword vectors

# #### Sample length of each word vector

# In[ ]:


model['coronavirus'].shape


# #### No of keywords * 100 dim vectors

# In[ ]:


words = [word for word in keywords]
X = model[words]
X.shape


# ## Reduce dimensionality from 100 to 2 for visualization

# In[ ]:


pca = PCA(n_components=2)
result = pca.fit_transform(X)
df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
df


# #### Add distance

# In[ ]:


df["Word"] = words
df["Distance"] = np.sqrt(df["Component 1"]**2 + df["Component 2"]**2)
df


# #### Let's add work frequencies as well

# In[ ]:


freq_list = []
for word in words:
    freq_list.append(model.wv.vocab[word].count)
df['frequency'] = freq_list
df


# In[ ]:


sns.scatterplot(data=df, x="Component 1", y="Component 2", 
                hue="Distance",size="frequency")


# ## 2-D visualization of Word2Vec embeddings
# Denoting frequency by size & distance by color

# In[ ]:


fig = px.scatter(df, x="Component 1", y="Component 2", text="Word", 
                 color="Distance", size="frequency", color_continuous_scale="agsunset")
fig.update_traces(textposition='top center')
fig.layout.xaxis.autorange = True
fig.data[0].marker.line.width = 1
fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
fig.update_layout(height=800, title_text="2D PCA of Word2Vec embeddings", 
                  template="plotly_white", paper_bgcolor="#f0f0f0")
fig.show()


# ## Let's pick a specific keyword

# #### Creating a function for this purpose

# In[ ]:


def pca_2d_similar(keyword):
    similar_words = model.wv.most_similar(keyword, topn=20)
    df_similar_words = pd.DataFrame(similar_words, columns = ['word', 'dist'])
    words = [word for word in df_similar_words['word'].tolist()]
    X = model[words]
    result = pca.fit_transform(X)
    df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
    df["Word"] = df_similar_words['word']
    word_emb = df[["Component 1", "Component 2"]].loc[0]
    df["Distance"] = np.sqrt((df["Component 1"] - word_emb[0])**2 + (df["Component 2"] - word_emb[1])**2)
    fig = px.scatter(df[2:], x="Component 1", y="Component 2", text="Word", color="Distance", color_continuous_scale="viridis",size="Distance")
    fig.update_traces(textposition='top center')
    fig.layout.xaxis.autorange = True
    fig.data[0].marker.line.width = 1
    fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
    fig.update_layout(height=800, title_text="2D PCA of words related to {}".format(keyword), template="plotly_white", paper_bgcolor="#f0f0f0")
    fig.show()
    
pca_2d_similar('antiviral')


# ## Further similarities

# In[ ]:


pca_2d_similar('antiretroviral')


# In[ ]:


pca_2d_similar('daas')


# In[ ]:


pca_2d_similar('cyclosporine')


# In[ ]:


pca_2d_similar('lamivudine')


# In[ ]:


pca_2d_similar('favipiravir')


# ## Let's look at cell structure

# In[ ]:


pca_2d_similar('rna')


# In[ ]:


pca_2d_similar('dna')


# In[ ]:





# In[ ]:




