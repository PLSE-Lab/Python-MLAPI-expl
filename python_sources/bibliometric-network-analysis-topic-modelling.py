#!/usr/bin/env python
# coding: utf-8

# # Bibliometric network analysis & topic modelling
# 
# Bibliometric data from academic databases can be used to find relationships between metadata (authors, titles, citations etc.) and discover dominant topics. In this kernel, we'll use the Metaknowledge package and an information science and bibliometrics dataset from Web of Science to perform network analysis and LDA topic modelling, along with visualizations. We'll try and answer the following questions:
# 
# 1. Which of the top authors are also top co-authors?
# 2. What does the co-authorship network look like?
# 3. What are the dominant topics that emerge from these academic papers?
# 
# The data and tutorials are available at https://github.com/mclevey/metaknowledge_article_supplement.

# In[ ]:


import os
import pandas as pd
import numpy as np
get_ipython().system('pip install metaknowledge # ensure Internet setting is "On" ')
import metaknowledge as mk
import networkx as nx
import community
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import ldamodel
from gensim.models import CoherenceModel 
import re
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


# Importing the information science and bibliometrics dataset
RC = mk.RecordCollection("../input/mk/raw_data/imetrics/")
len(RC)


# In[ ]:


RC


# The data is currently stored as a RecordCollection object and must be converted into a dataframe if we want to see its contents. We can do this in two ways: with Pandas or with Metaknowledge's makeDict() function.

# In[ ]:


# Saving the dataset as a csv file
RC.writeCSV("records.csv")
# Reading in the data as a Pandas dataframe
data = pd.read_csv("records.csv")
data.head(3)


# In[ ]:


# Saving the data as a dataframe using mk's makeDict()
data2 = pd.DataFrame(RC.makeDict())
data2.head(3)


# The makeDict() function gets rid of the id column and rearranges some columns, but the data is essentially the same. The two-letter variable names are tags used by Web of Science. See their description here - https://images.webofknowledge.com/images/help/WOK/hs_alldb_fieldtags.html
# 
# Metaknowledge also provides the handy function glimpse() to view the top authors, journals and citations in the database.

# In[ ]:


# Printing basic statistics about the data
print(RC.glimpse())


# ## Network Analysis
# 
# Network analysis lets us see the relationships, distances and co-occurrences between the nodes in a network. Here's how we can conduct a network analysis of co-authors in this dataset.

# In[ ]:


# Generating the co-author network 
coauth_net = RC.networkCoAuthor()
coauth_net


# In[ ]:


# Printing the network stats
print(mk.graphStats(coauth_net))


# There are 10104 nodes (authors) in the network who are connected by 15507 edges. Of these authors, 1111 are isolates (unconnected to others).
# 
# Next, we will drop self-loops and any authors with fewer than 2 edges (co-authors). For our analysis we will extract the "giant component", which is the largest subgraph of connected nodes in a network graph. The giant component typically contains a significant proportion of the nodes in the network. We'll use Python's networkx package for this and subsequent tasks.

# In[ ]:


mk.dropEdges(coauth_net, minWeight = 2, dropSelfLoops = True)
giant_coauth = max(nx.connected_component_subgraphs(coauth_net), key = len)
print(mk.graphStats(giant_coauth))


# We are left with 265 authors, all of whom have at least two co-authors. We can see the graph density has gone up because of our filtering criteria.
# 
# Centrality is a key concept in network analysis. The degree, closeness, betweenness and eigenvector centralities tell us which nodes (authors) are the most important. These are calculated from the number of links to other nodes, the length of the paths to other nodes, the number of times the node acts as a bridge along the shortest path between other nodes, and the relative influence of the node, respectively.
# 
# Let's compute the centrality scores in our co-author graph.

# In[ ]:


# Computing centrality scores
deg = nx.degree_centrality(giant_coauth)
clo = nx.closeness_centrality(giant_coauth)
bet = nx.betweenness_centrality(giant_coauth)
eig = nx.eigenvector_centrality(giant_coauth)

# Saving the scores as a dataframe
cent_df = pd.DataFrame.from_dict([deg, clo, bet, eig])
cent_df = pd.DataFrame.transpose(cent_df)
cent_df.columns = ["degree", "closeness", "betweenness", "eigenvector"]

# Printing the top 10 co-authors by degree centrality score
cent_df.sort_values("degree", ascending = False)[:10]


# In[ ]:


# Visualizing the top 10 co-authors by degree centrality score
sns.set(font_scale=.75)
cent_df_d10 = cent_df.sort_values('degree', ascending = False)[:10]
cent_df_d10.index.name = "author"
cent_df_d10.reset_index(inplace=True)
print()
plt.figure(figsize=(10,7))
ax = sns.barplot(y = "author", x = "degree", data = cent_df_d10, palette = "Set2");
ax.set_alpha(0.8)
ax.set_title("Top 10 authors in co-author graph", fontsize = 18)
ax.set_ylabel("Authors", fontsize=14);
ax.set_xlabel("Degree centrality", fontsize=14);
ax.tick_params(axis = 'both', which = 'major', labelsize = 14)


# The top 3 authors in the co-author network are the same as the top 3 authors in the original Record Collection. However, there are 5 authors in the original top 10 who are missing from the top 10 co-authors. 
# 
# Let's calculate the solo authorship and co-authorship rates for these 10 authors.

# In[ ]:


solo = []
co = []
for i in cent_df_d10["author"]:
    # Calculate solo authorship rate
    so = np.round((len(data[(data['AF'].str.contains(i)) & (data["num-Authors"] == 1)])) / (len(data[data['AF'].str.contains(i)])), decimals = 2)
    solo.append(so)
    # Calculate co-authorship rate
    co.append(1-so)
print(solo, co)


# In[ ]:


# Create top 10 authors dataframe
authors = pd.DataFrame(zip(solo, co), columns = ["solo", "coauthor"])
authors["author"] = cent_df_d10["author"]
# Rearrange columns
authors = authors[["author", "solo", "coauthor"]]
authors


# In[ ]:


sns.set(rc={'figure.figsize':(8,7)})
fig = sns.swarmplot(x = "solo", y = "coauthor", hue = "author", data = authors, s = 12, palette = "Paired")
plt.title("Solo vs. Co-authorship rate: Top 10 Authors", fontsize = 16)
plt.xlabel("Solo authorship rate")
plt.ylabel("Co-authorship rate")
plt.show()


# Among the top 10 co-authors by degree centrality, Bornmann and Leydesdorff have the highest solo authorship rates while de Moya-Anegon and Wang have not authored any papers solo. 
# 
# Next, we will visualize the co-author network and then perform community detection using the Louvain method. The Louvain method maximizes a modularity score for each community or group of authors, which quantifies how "good" the communities are. (It does this by evaluating how much more densely connected the nodes within a community are, compared to how connected they would be in a random network.)
# 
# Network visualizations can be difficult and confusing. There are several possible layouts, but we'll use the "spring layout" which results in a more aesthetic graph.

# In[ ]:


# Visualizing the co-author network
plt.figure(figsize = (10, 7))
size = [2000 * eig[node] for node in giant_coauth]
nx.draw_spring(giant_coauth, node_size = size, with_labels = True, font_size = 5,
               node_color = "#FFFFFF", edge_color = "#D4D5CE", alpha = .95)


# In[ ]:


# Community detection
partition = community.best_partition(giant_coauth) 
modularity = community.modularity(partition, giant_coauth)
print("Modularity:", modularity)


# In[ ]:


# Visualizing the communities
# Generates a different graph each time
plt.figure(figsize = (10, 7))
colors = [partition[n] for n in giant_coauth.nodes()]
my_colors = plt.cm.Set2 
nx.draw(giant_coauth, node_color=colors, cmap = my_colors, edge_color = "#D4D5CE")


# The colourful visualization above reveals communities of co-authors in the network. Note that these are not fixed positions or shapes of the communities. The graph can look different each time.
# 
# We've analysed relationships between authors by computing and visualizing centralities in the co-author network. Next, we'll use NLTK and Gensim to preprocess and performing topic modelling on the academic journal data.
# 
# ## Topic Modelling
# 
# The Metaknowledge function forNLP() creates a Pandas-friendly dictionary where each row is a record from the RecordCollection, and the columns contain textual data (id, title, publication year, keywords and the abstract). Its results are *not* reproducible - the records appear to be shuffled each time.
# 
# Before we proceed, here's an overview of the ideas behind topic modelling with LDA:
# 
# - Latent Dirichlet Allocation is a probabilistic model that generates topic-document and word-topic probability distributions. Topics are themes that occur in documents, and each word in a document belongs to some topic. 
# - The word "latent" refers to the topics, which are hidden, while Dirichlet refers to the probability distribution over 'k' number of topics. Allocation refers to the process of assigning topics to documents and words to topics. 
# - This allocation is an iterative process in which an initial topic distribution is assumed, and the weights for each word are updated till the model converges.
# - LDA tries to allocate words to a fewer number of topics in each document, and tries to assign high probabilities to only a few words within each topic. These are conflicting goals and the tradeoff results in groups of co-occuring words.
# - The model produces topics by inferring the conditional or posterior distribution over the latent variables, given the observed words and parameters.

# In[ ]:


# Transform the record collection into a format for use with natural language processing applications
data = RC.forNLP("topic_model.csv", lower=True, removeNumbers=True, removeNonWords=True, removeWhitespace=True)

# Convert the raw text into a list.
docs = data['abstract']
docs


# Since the data is in text form, we'll do some tokenizing and lemmatizing, remove stop words and create a term-document matrix to bring it into the right format for topic modelling.

# In[ ]:


# Defining a function to clean the text
def clean(docs):
    # Insert function for preprocessing the text
    def sent_to_words(sentences):
        for sentence in sentences:
            yield (simple_preprocess(str(sentence), deacc = True))
    # Tokenize the text
    tokens = sent_to_words(docs)
    # Create stopwords set
    stop = set(stopwords.words("english"))
    # Create lemmatizer
    lmtzr = WordNetLemmatizer()
    # Remove stopwords from text
    tokens_stopped = [[word for word in post if word not in stop] for post in tokens]
    # Lemmatize text
    tokens_cleaned = [[lmtzr.lemmatize(word) for word in post] for post in tokens_stopped]
    # Return cleaned text
    return tokens_cleaned

# Cleaning up the raw documents
cleaned_docs = clean(docs)
cleaned_docs


# In[ ]:


# Creating a dictionary
id2word = corpora.Dictionary(cleaned_docs)
print(id2word)


# There are 21943 unique words in the text. We'll filter out infrequent and overly frequent words from the dictionary, as this can improve the topic model.

# In[ ]:


# Filtering infrequent and over frequent words
id2word.filter_extremes(no_below=5, no_above=0.5)
# Creating a document-term matrix
corpus = [id2word.doc2bow(doc) for doc in cleaned_docs]


# In[ ]:


# Building an LDA model with 5 topics
model = ldamodel.LdaModel(corpus = corpus, num_topics = 5, id2word = id2word, 
                              passes = 10, update_every = 1, chunksize = 1000, per_word_topics = True, random_state = 1)
# Printing the topic-word distributions
pprint(model.print_topics())


# The topics look well-defined and cohesive (remember we are using an Information Science and Bibliometrics dataset). Finally, we'll create an interactive visualization of the model that shows inter-topic distances and topic-word distributions.

# In[ ]:


pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(model, corpus, id2word, mds = "tsne")
vis

