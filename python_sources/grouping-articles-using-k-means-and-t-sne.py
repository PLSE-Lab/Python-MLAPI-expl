#!/usr/bin/env python
# coding: utf-8

# # Clustering German Articles using Word2Vec
# 
# ## Word2Vec
# 
# Word2Vec is a method to represent words in a numerical - vector format such that words that are closely related to each other are close to each other in numeric vector space. This method was developed by Thomas Mikolov in 2013 at Google.
# 
# Each word in the corpus is modeled against surrounding words, in such a way that the surrounding words get maximum probabilities. The mapping that allows this to happen , becomes the word2vec representation of the word. The number of surrounding words can be chosen through a model parameter called "window size". The length of the vector representation is chosen using the parameter 'size'.
# 
# In this notebook, the library gensim is used to construct the word2vec models
# 
# ## Loading the library and getting the data
# 

# In[ ]:




import pandas as pd
import numpy as np
import os
import re
import gensim
import spacy
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("german")



data_test = pd.read_csv('../input/10k-german-news-articles/Articles.csv')
data_test.head()


# In[ ]:



body_new = [re.sub('<[^<]+?>|&amp', '', word) for word in data_test['Body'].values]
data_test['Body'] = body_new
#! python -m spacy download de_core_news_sm


# To tackle stopwords in German, the spaCy module is used. The spaCy module consists of information regarding stopwords for different languages. The command `! python -m spacy download de_core_news_sm` enables the download of German- related module

# In[ ]:


get_ipython().system(' python -m spacy download de_core_news_sm')
import spacy
import de_core_news_sm
nlp = de_core_news_sm.load()


#nlp = spacy.load('de_core_news_sm')


# ## Dropping NA values and Duplicate Articles
# 
# From the pulled data, rows that contain NA values are dropped. Duplicate rows are also dropped. After this, the columns "Headline" & "Body" are chosen. Each article is indexed with the row number it belongs in.

# In[ ]:


## Drop Duplicates and NA values

data_test_clean = data_test.dropna()
data_test_clean = data_test_clean.drop_duplicates()


# In[ ]:


article_data = data_test_clean[['ID_Article','Title','Body']].drop_duplicates()
article_data.shape


# After dropping rows that contain NA values and rows that have duplicated values, redundant words such as punctuation marks and stopwords are removed. As the texts are in German, the stopword corpus for German is used from the `spacy` library. After this preprocessing, the words are converted to a list of words to be fed into the model.

# In[ ]:



## Convert the body text into a series of words to be fed into the model

def text_clean_tokenize(article_data):
    
    review_lines = list()

    lines = article_data['Body'].values.astype(str).tolist()

    for line in lines:
        tokens = word_tokenize(line)
        tokens = [w.lower() for w in tokens]
        table = str.maketrans('','',string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words('german'))
        words = [w for w in words if not w in stop_words]
        words = [stemmer.stem(w) for w in words]

        review_lines.append(words)
    return(review_lines)
    
    
review_lines = text_clean_tokenize(article_data)


# In[ ]:


## Building the word2vec model

model =  gensim.models.Word2Vec(sentences = review_lines,
                               size=100,
                               window=2,
                               workers=4,
                               min_count=2,
                               seed=42,
                               iter= 50)

model.save("word2vec.model")


# The word2vec representation of each word is used to build the representations for articles. The constitutent words's word2vec representations are rolled up and averaged. This averaged vector would be the numerical vector representation of the article in question.

# In[ ]:


import spacy

def tokenize(sent):
    doc = nlp.tokenizer(sent)
    return [token.lower_ for token in doc if not token.is_punct]

new_df = (article_data['Body'].apply(tokenize).apply(pd.Series))

new_df = new_df.stack()
new_df = (new_df.reset_index(level=0)
                .set_index('level_0')
                .rename(columns={0: 'word'}))

new_df = new_df.join(article_data.drop('Body', 1), how='left')

new_df = new_df[['word','ID_Article','Title']]
word_list = list(model.wv.vocab)
vectors = model.wv[word_list]
vectors_df = pd.DataFrame(vectors)
vectors_df['word'] = word_list
merged_frame = pd.merge(vectors_df, new_df, on='word')
merged_frame_rolled_up = merged_frame.drop('word',axis=1).groupby(['ID_Article','Title']).mean().reset_index()
del merged_frame
del new_df
del vectors


# In[ ]:


merged_frame_rolled_up.columns


# ## K- Means Clustering
# 
# As the data set is un-labelled, its our job to find structure and label it accordingly. For this purpose, the K - Means clustering algorithm is used. 'K' or the number of clusters is chosen dynamically by looking K for which the difference between the first and second differential of change in distortion is maximum. This idea of dynamically choosing K can be found in this article. https://www.datasciencecentral.com/profiles/blogs/how-to-automatically-determine-the-number-of-clusters-in-your-dat
# 

# In[ ]:



from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
data_subset = merged_frame_rolled_up.drop(['ID_Article','Title'],axis=1).values

from sklearn.cluster import KMeans
number_of_clusters = range(1,20,5)
kmeans = [KMeans(n_clusters=i,max_iter=5,init='k-means++') for i in number_of_clusters]
score = [kmeans[i].fit(data_subset).inertia_ for i in range(len(kmeans))]
tmp =0
best_k = 0
value_all = 0
diff1 = []
for i in range(len(score)-1):
    
    scores = (score[i] - score[i+1])
    diff1.append(scores)
    
diff2=[]
for i in range(len(diff1)-1):
    difference = diff1[i] - diff1[i+1]
    diff2.append(difference)
diff2.insert(0, 0) 
diff3 = [i-j for i,j in zip(diff2,diff1)]

m = max(i for i in diff3)
best_k = number_of_clusters[diff3.index(m)]
print(best_k)

kmeans = KMeans(n_clusters=best_k,init='k-means++',max_iter=5).fit(data_subset)
labels = kmeans.labels_
merged_frame_rolled_up['labels'] = labels




# ## Visualizing data
# 
# After each data point is assigned a cluster, the next natural step is to visualize it. But there is a problem. The data we have has over 100 columns. We will need to reduce the number of columns to at most 3 to visualize it. Here to do that, we use t-SNE, which helps keep closer points together and farther points far from each other in lower dimensions. The visualization was made possible by using this kernel. https://www.kaggle.com/maksimeren/covid-19-literature-clustering#Dimensionality-Reduction-with-t-SNE
# 

# In[ ]:


tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=3000,init='pca',random_state=42)
tsne_results = tsne.fit_transform(data_subset)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

merged_frame_rolled_up['tsne-2d-one'] = tsne_results[:,0]
merged_frame_rolled_up['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue = 'labels',
    palette=sns.color_palette("hls", len(merged_frame_rolled_up['labels'].unique())),
    data=merged_frame_rolled_up,
    legend="full",
    alpha=0.3
)


# ## Interactive Visualization using Bokeh

# In[ ]:


## Interactive t-SNE mapping using https://www.kaggle.com/maksimeren/covid-19-literature-clustering

from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.palettes import Category20
from bokeh.transform import linear_cmap
from bokeh.io import output_file, show
from bokeh.transform import transform
from bokeh.io import output_notebook
from bokeh.plotting import figure

output_notebook()

# data sources
source = ColumnDataSource(data=dict(
    x= merged_frame_rolled_up['tsne-2d-one'].values, 
    y= merged_frame_rolled_up['tsne-2d-two'].values, 
    desc= merged_frame_rolled_up['labels'].values, 
    titles= merged_frame_rolled_up['Title'].values
    ))

# hover over information
hover = HoverTool(tooltips=[
    ("Title", "@titles")
])
# map colors
mapper = linear_cmap(field_name='desc', 
                     palette=Category20[len(merged_frame_rolled_up['labels'].unique())],
                     low=min(merged_frame_rolled_up['labels'].values) ,high=max(merged_frame_rolled_up['labels'].values))
#prepare the figure
p = figure(plot_width=1000, plot_height=1000, 
           tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset'], 
           title="Clustering German Articles Based on K Means and Word2Vec", 
           toolbar_location="right")
# plot
p.scatter('x', 'y', size=5, 
          source=source,
          fill_color=mapper,
          line_alpha=0.3,
          line_color="black")

# output_file('plots/t-sne_covid-19_interactive.html')
show(p)


# In[ ]:





# 
