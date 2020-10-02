#!/usr/bin/env python
# coding: utf-8

# # Analysing text similarity using spaCy, networkX 
# 
# This notebook demonstrates one way of using spaCy to conduct a rapid thematic analysis of a small corpus of comments, and introduces some unusual network visualisations.
# Topics include: 
# * [spaCy](https://spacy.io/) - an open source NLP library, 
# * word vectors, and
# * networkX - an open source network (graph) analysis and visualisation library. 
# 
# The notebook is partly a reminder for myself on just how (well) these techniques work, but I hope that others find it useful. I'll continue to update it with more techniques over the coming weeks.
# If you have any suggestions, feel free to make them in the comments, fork the notebook etc. I'm keen to exchange tips and tricks. 
# 

# # Plan

# * load a representative set of tweets
# * demonstrate some basic spaCy features
# * test its similarity metrics
# * build a graph data structure for storing (n * n-1) / 2 similarity results
# * visualise the clusters of most-similar items in the data
# * plan the next steps

# In[ ]:


import pandas as pd
import spacy
import networkx as nx                        # a really useful network analysis library
import matplotlib.pyplot as plt
# from networkx.algorithms import community   # not used, yet... 
import datetime                              # access to %%time, for timing individual notebook cells
import os


# This next step load the spaCy language model. It generally takes about 13s to load this 'large' model.

# In[ ]:


nlp = spacy.load('en_core_web_lg')           # A more detailed model (with higher-dimension word vectors) - 13s to load, normally 
#nlp = spacy.load('en_core_web_md')           # a smaller model, e.g. for testing


# In[ ]:


plt.rcParams['figure.figsize'] = [10, 10]  # makes the output plots large enough to be useful


# ## Data

# In[ ]:


rowlimit = 500              # this limits the tweets to a manageable number
data = pd.read_csv('../input/ExtractedTweets.csv', nrows = rowlimit)
data.shape


# In[ ]:


data.head(6)


# ## Using spaCy to parse the tweets.

# N.B. this next step can take a while - e.g. 14 mins, for the full set - but only 5s for 500 rows.
# 
# (based on https://stackoverflow.com/questions/44395656/applying-spacy-parser-to-pandas-dataframe-w-multiprocessing)...

# In[ ]:


tokens = []
lemma = []
pos = []
parsed_doc = [] 
col_to_parse = 'Tweet'

for doc in nlp.pipe(data[col_to_parse].astype('unicode').values, batch_size=50,
                        n_threads=3):
    if doc.is_parsed:
        parsed_doc.append(doc)
        tokens.append([n.text for n in doc])
        lemma.append([n.lemma_ for n in doc])
        pos.append([n.pos_ for n in doc])
    else:
        # We want to make sure that the lists of parsed results have the
        # same number of entries of the original Dataframe, so add some blanks in case the parse fails
        tokens.append(None)
        lemma.append(None)
        pos.append(None)


data['parsed_doc'] = parsed_doc
data['comment_tokens'] = tokens
data['comment_lemma'] = lemma
data['pos_pos'] = pos


# ## Basic checks of the parsed data

# In[ ]:


data.head(8)


# In[ ]:


data.Tweet[0]


# In[ ]:


data.Tweet[1]


# In[ ]:


data.Tweet[10]


# ## Removing stopwords

# We could reduce increase the signal:noise ratio in these texts by removing some of the more common words (or *stopwords*). By removing these from the tweets, we would prevent them from influencing the analysis of whether two tweets are similar. I'm not addressing this is the notebook yet, but I will come back to it later. For now, let's just look at what words are included in spaCy's stopword list.

# In[ ]:


stop_words = spacy.lang.en.stop_words.STOP_WORDS
print('Number of stopwords: %d' % len(stop_words))
print(list(stop_words))


# ## Testing spaCy's similarity function

# In[ ]:


print(data['parsed_doc'][0].similarity(data['parsed_doc'][1]))
print(data['parsed_doc'][0].similarity(data['parsed_doc'][10]))
print(data['parsed_doc'][1].similarity(data['parsed_doc'][10]))


# If you've limited the rows imported, then you may only have Democrat tweets (which occur first in the list).

# In[ ]:


data.Party.unique()


# In[ ]:


world_data = data
#world_data = data[data.Party == 'Democrat']      # or use either of these, if you want to see tweets from only one party
#world_data = data[data.Party == 'Republican']


# In[ ]:


# takes 1s for 500 nodes - but of course this won't scale linearly!                              
raw_G = nx.Graph() # undirected
n = 0

for i in world_data['parsed_doc']:        # sure, it's inefficient, but it will do
    for j in world_data['parsed_doc']:
        if i != j:
            if not (raw_G.has_edge(j, i)):
                sim = i.similarity(j)
                raw_G.add_edge(i, j, weight = sim)
                n = n + 1

print(raw_G.number_of_nodes(), "nodes, and", raw_G.number_of_edges(), "edges created.")


# In[ ]:


edges_to_kill = []
min_wt = 0.94      # this is our cutoff value for a minimum edge-weight 

for n, nbrs in raw_G.adj.items():
    #print("\nProcessing origin-node:", n, "... ")
    for nbr, eattr in nbrs.items():
        # remove edges below a certain weight
        data = eattr['weight']
        if data < min_wt: 
            # print('(%.3f)' % (data))  
            # print('(%d, %d, %.3f)' % (n, nbr, data))  
            #print("\nNode: ", n, "\n <-", data, "-> ", "\nNeighbour: ", nbr)
            edges_to_kill.append((n, nbr)) 
            
print("\n", len(edges_to_kill) / 2, "edges to kill (of", raw_G.number_of_edges(), "), before de-duplicating")


# In[ ]:


for u, v in edges_to_kill:
    if raw_G.has_edge(u, v):   # catches (e.g.) those edges where we've removed them using reverse ... (v, u)
        raw_G.remove_edge(u, v)


# In[ ]:


strong_G = raw_G
print(strong_G.number_of_edges())


# We should now have a clean graph of only hi-similarity edges.

# ## Visualising the selected edges

# NetworkX has several useful layouts implemented, but you can't beat a good spring-embedding layour (a kind of [force-directed graph](https://en.wikipedia.org/wiki/Force-directed_graph_drawing)).
# In graph terminology, what we see is:
# * a single large [component](https://en.wikipedia.org/wiki/Connected_component_(graph_theory)) at the centre,
# * with several [pendants](https://proofwiki.org/wiki/Definition:Pendant_Vertex) visible at the edges;
# * several smaller components; and 
# * a peripheral cloud of [isolates](http://mathonline.wikidot.com/isolated-vertices-leaves-and-pendant-edges)
# 
# Force-directed graphs are a very intuitive, satisfying, and efficient way to lay out network diagrams. Essentially, every node exerts a repulsive force on every other node. Simultaneously, every connected pair of nodes attract each other. The layout algorithm iterates, finding a layout that balances these forces.

# In[ ]:


nx.draw(strong_G, node_size=20, edge_color='gray')


# Visualising the whole graph, but only those links of weights above a certain cutoff, allows us to get a feel for a good cutoff level to use when visualising the structure. Having filtered out these lower-weighted links, we can clean up the graph by removing the isolates. This will enable the layout engine to show us more of the structure of the components.

# In[ ]:


strong_G.remove_nodes_from(list(nx.isolates(strong_G)))


# We can also tweak the layout algorithm. By, for example, changing the ideal distance at which the repulsive and attractive forces are in equilibrium. There's a good description of these forces [here](https://schneide.blog/tag/fruchterman-reingold/). This value interacts with the number of `iterations` in surprising ways.

# In[ ]:


from math import sqrt
count = strong_G.number_of_nodes()
equilibrium = 10 / sqrt(count)    # default for this is 1/sqrt(n), but this will 'blow out' the layout for better visibility
pos = nx.fruchterman_reingold_layout(strong_G, k=equilibrium, iterations=300)
nx.draw(strong_G, pos=pos, node_size=10, edge_color='gray')


# Of course, we can specify the layout we want to use, change colours, sizes, etc. The following cell adds the text of the tweets - which can make the layout hard to read.

# In[ ]:


plt.rcParams['figure.figsize'] = [16, 9]  # a better aspect ratio for labelled nodes

nx.draw(strong_G, pos, font_size=3, node_size=50, edge_color='gray', with_labels=False)
for p in pos:  # raise positions of the labels, relative to the nodes
    pos[p][1] -= 0.03
nx.draw_networkx_labels(strong_G, pos, font_size=8, font_color='k')

plt.show()


# ## Next Steps

# I hope this notebook was useful. Next:
# * I'd like to apply some keyword extraction to the tweets, to make this visualisation more useful;
# * there'll be some topic identification using gensim's implementation of LDA;
# * some more intelligent parameterisation of variables, such as allowing the minimum similarity cut-off to account for network size;
# * I'd like to apply a smarter similarity cut-off, such as Vladimir Batagelj's '[vertex islands](http://vlado.fmf.uni-lj.si/pub/networks/doc/mix/islands.pdf)' technique; and
# * I should really apply TF-IDF, if only just to see how it compares to other keyword extraction techniques.
