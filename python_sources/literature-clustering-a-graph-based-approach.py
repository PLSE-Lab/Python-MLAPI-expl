#!/usr/bin/env python
# coding: utf-8

# # Housekeeping
# First set some filenames. It just makes the code look cleaner. Forgive me for being used to multi-file environments where you wouldn't have to see this mess.

# In[ ]:


import os

# File locations
JSON_DATA_DIR = '../input/CORD-19-research-challenge/'
HOME = '/kaggle/working/'

#   JSON data
JSON_DATA_DIRS = ['biorxiv_medrxiv/'*2 + 'pdf_json/', 'comm_use_subset/'*2 + 'pdf_json/']
DICTS = HOME + 'dictionaries/'
try:
    os.mkdir(DICTS)
except FileExistsError:
    pass

#   Dict data
WORD_DATA_DIR = HOME + 'dictionaries/'
CORPUS_F = WORD_DATA_DIR + 'corpus.data'

#   Graph data
GRAPH_FILE = HOME + 'graph.npz'
NODE_EMBEDDINGS = HOME + 'node_embeddings.model'

# File naming functions
F_TO_JSON = lambda x : (x.split('/')[-1]).split('.')[0] + '.json'
F_TO_DICT = lambda x : (x.split('/')[-1]).split('.')[0] + '.data'
F_TO_HASH = lambda x : F_TO_JSON(x).split('.')[0]

# Lists of all data files
#   JSON data
JSON_FILES = []
for d in JSON_DATA_DIRS:
    JSON_FILES += [JSON_DATA_DIR + d + f for f in os.listdir(JSON_DATA_DIR + d)]
#   Dict data
WORD_DATA_FILES = [WORD_DATA_DIR + F_TO_DICT(f) for f in JSON_FILES]

NUM_DOCS = len(JSON_FILES)

HASH_IDX = {F_TO_HASH(JSON_FILES[i]): i for i in range(len(JSON_FILES))}


# Added some stopwords specific to journal papers
# or that slipped through NLTK's default list    
CUSTOM_STOPWORDS = [
    "n't", 
    "'m", 
    "'re", 
    "'s", 
    "nt", 
    "may",
    "also",
    "fig",
    "http"
]


# # Pre-processing
# ### Word Frequency Dictionaries
# Now we can get down to business. First we generate the dictionaries of words contained in the data. The pipeline for this process is pretty standard: tokenize, lemmatize, & remove stopwords. These dictionaries on their own may be enough to make some vectors an do KNN analysis clustering, but we're using a graph-based approach, so when we're done here we have one more step. 

# In[ ]:


import os
import nltk
import json
import pickle
import random

from tqdm.notebook import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

LEMMATIZER = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english') + CUSTOM_STOPWORDS)

def pipeline(text):
    '''
    Processes raw text to remove all stopwords and lemmatize
    '''
    t = nltk.word_tokenize(text)
    t = [LEMMATIZER.lemmatize(w) for w in t]
    t = [w for w in t if w not in STOPWORDS]
    t = [w for w in t if len(w) > 1]
    return t


def run(documents, save=True):
    ''' 
    Given several documents, generates 1 dict of all words
    used in the whole corpus, and a dict for each document.
    The dicts map the word to its frequency
    '''
    progress = tqdm(total=NUM_DOCS, desc='JSON Files parsed:')
    corpus = {}

    for document in documents:
        with open(document, 'r') as f:
            schema = json.loads(f.read().lower())
            paper_id = schema['paper_id']

            text = schema['metadata']['title'] + ' '

            '''
            # Naive (?) approach: use paper text to build graphs
            paragraphs = schema['body_text']
            for p in paragraphs:
                text += p['text']
            '''
            
            '''
            # Use only titles to build graph
            for ref in schema['bib_entries'].values():
                text += ref['title'] + ' '
            '''
              
            # Use only abstract and title to build graph
            for par in schema['abstract']:
                text += par['text'] + ' '
            

        text = pipeline(text)
        
        doc_dict = {}
        for word in text:
            # Assume it's already accounted for in corpus
            if word in doc_dict:
                doc_dict[word] += 1
                corpus[word]['count'] += 1

            else:
                doc_dict[word] = 1
                
                # Make sure to add this paper to the corpus to make building
                # the graph eaiser later on
                if word in corpus:
                    corpus[word]['count'] += 1
                    corpus[word]['papers'].add(paper_id)
                else:
                    corpus[word] = {'count': 1, 'papers': {paper_id}}

        if save:
            pickle.dump(
                doc_dict, 
                open(DICTS+F_TO_DICT(document), 'wb+'), 
                protocol=pickle.HIGHEST_PROTOCOL
            )
        
        progress.update()

    if save:
        pickle.dump(
            corpus, 
            open(CORPUS_F, 'wb+'),
            protocol=pickle.HIGHEST_PROTOCOL
        )
    
    return corpus

def runall():
    # I'm sure there's a smarter way to do this but who cares
    run(JSON_FILES)

def test(num_docs=10):
    test_docs = []
    for i in range(num_docs):
        test_docs.append(random.choice(JSON_FILES))

    return run(test_docs, save=False)
    
runall()


# # Generating vectors
# ### Building a Graph
# Nice. It looks like the frequency dictionaries are done. Now we can use the frequency data we've collected to build a graph. Nothing fancy, just an undirected, homogeneous graph with weighted edges. Nodes are papers, and edges between them are shared words weighted by TF-IDF score. For simplicity, we merge edges together by summing their weights. To filter out any noise we missed when removing stop words, we don't consider edges with lower TF-IDF than some preset threshold. We have selected 10 for this threshold.

# In[ ]:


import os
import json 
import pickle
import random
import numpy as np

from scipy.sparse import csr_matrix, save_npz 
from tqdm.notebook import tqdm
from math import log

TF_IDF_THRESHOLD = 10

def tf_idf(tf, doc_count):
    idf = log(NUM_DOCS/doc_count)
    return tf*idf

def build_graph(documents):
    # Undirected, regular old graph
    g = np.zeros((NUM_DOCS, NUM_DOCS))
    corpus = pickle.load(open(CORPUS_F, 'rb'))
    
    # Represent graph as a sparse CSR matrix as row slices are important
    # but most papers have very few neighbors
    row = [0]
    cols = []
    data = []
    
    last_idx = 0
    progress = tqdm(total=NUM_DOCS, desc='Number of nodes added:')
    for node_id in range(len(documents)):
        doc_dict = pickle.load(open(WORD_DATA_FILES[node_id], 'rb'))
        col = []
        cdata = []
        
        # Link with all papers that share significant words
        for word, count in doc_dict.items():
            thresh = tf_idf(count, len(corpus[word]['papers']))
            
            if thresh > TF_IDF_THRESHOLD:
                for paper in corpus[word]['papers']:
                    neigh_id = HASH_IDX[paper]
                    
                    # Prevent self-loops
                    if neigh_id == node_id:
                        continue
                    
                    # Edge weights are the sum of each tf-idf score of shared words
                    # This is functionally equivilant to using a multi-graph
                    # as later on, we do random walks based on these weights
                    # so P(B|A) is the same in both cases
                    if neigh_id in col:
                        cdata[col.index(neigh_id)] += thresh
                    else:
                        col.append(neigh_id)
                        cdata.append(thresh)
        
        # Update CSR Matrix stuff           
        last_idx += len(col)     
        row.append(last_idx)
        cols += col
        data += cdata
        
        progress.update()
    
    print("Building matrix from parsed data")
    g = csr_matrix((data, cols, row), shape=(len(documents), len(documents)))
    save_npz(GRAPH_FILE, g)
    return g
    
def test(num_nodes=10):
    docs = []
    for i in range(num_nodes):
        docs.append(random.choice(WORD_DATA_FILES))
    
    g = build_graph(docs)
    return g
    
def run():
    return build_graph(WORD_DATA_FILES)

if __name__ == '__main__':
    run()


# ### Generate Node Embeddings
# Last, we need to have some vectors to work with. In order to do this, we use the node2vec technique with the skip-gram model. However, we weight the random walks based on edge weights, so the walk is more likely to go to similar nodes. We use 200 walks of length 3 (the source node plus 3 more) to generate the "sentences" that we feed into Word2Vec. By using a random-walk approach, this is like an approximation of KNN clustering. Rather than checking the distance from all other vectors, we simply classify nodes by what nodes they are strongly connected to. 

# In[ ]:


import random
import numpy as np

from joblib import Parallel, delayed
from tqdm.notebook import tqdm 
from gensim.models import Word2Vec
from scipy.sparse import load_npz

# Model parameters
NUM_WALKS = 200
WALK_LEN = 3

# W2V params
NUM_WORKERS = 4
W2V_PARAMS = {
    'size': 256,
    'workers': NUM_WORKERS,
    'sg': 1,
}

def generate_walks(num_walks, walk_len, g, starter):
    '''
    Generate random walks on graph for use in skipgram
    '''
    
    # Allow random walks to be generated in parallel given list of nodes
    # for each worker thread to explore
    walks = []
    
    # Can't do much about nodes that have no neighbors
    if g[starter].data.shape[0] == 0:
        return [[str(starter)]]
    
    for _ in range(num_walks):
        walk = [str(starter)]
        n = starter
        
        # Random walk with weights based on tf-idf score
        for __ in range(walk_len):
            # Pick a node weighted randomly from neighbors
            # Stop walk if hit a dead end
            if g[n].data.shape[0] == 0:
                break
            
            next_node = random.choices(
                g[n].indices,
                weights=g[n].data
            )[0]  
            
            walk.append(str(next_node))
            n = next_node 
                
        walks.append(walk)
    
    return walks

def generate_walks_parallel(g, walk_len, num_walks, workers=1):
    '''
    Distributes nodes needing embeddings across all CPUs 
    Because this is just many threads reading one datastructure this
    is an embarrasingly parallel task
    '''
    flatten = lambda l : [item for sublist in l for item in sublist]     
        
    print('Executing tasks')
    # Tell each worker to generate walks on a subset of
    # nodes in the graph
    walk_results = Parallel(n_jobs=workers, prefer='processes')(
        delayed(generate_walks)(
            num_walks, 
            walk_len,
            g,
            node
        ) 
        for node in tqdm(range(NUM_DOCS), desc='Walks generated:')
    )
    
    return flatten(walk_results)


def embed_walks(walks, params, fname):
    '''
    Sends walks to Word2Vec for embeddings
    '''
    print('Embedding walks...')
    model = Word2Vec(walks, **params)
    model.save(fname)
    return model.wv.vectors

def load_embeddings(fname=NODE_EMBEDDINGS):
    return Word2Vec.load(fname).wv.vectors


fname = NODE_EMBEDDINGS

print('Loading graph')
g = load_npz(GRAPH_FILE)

print('Generating walks')
walks = generate_walks_parallel(g, WALK_LEN, NUM_WALKS, workers=NUM_WORKERS)
    
print('Embedding nodes')
embed_walks(walks, W2V_PARAMS, fname)


# # Clustering 
# Finally, we cluster the papers. We have selected 25 classes, but this is easilly changable. Visualizing the data is a little trickier, but we use t-SNE to attempt to make 2D visualization possible. Without classes, it is difficult to say for sure if this technique worked, but from the titles of the papers near each other, it appears the clustering works rather nicely

# First, a bit of housekeeping. We need to load the metadata so we can actually map the file hashes back to the titles of the papers

# In[ ]:


import pandas as pd
META = JSON_DATA_DIR + 'metadata.csv'
df = pd.read_csv(META)


# Next, we cluster using the AgglomerativeClustering method, out of the box. This appears to work well enough. We use this to predict 25 distinct clusters, then run the embeddings through t-SNE to better visualize the 256-dimensional vectors

# In[ ]:


from gensim.models import Word2Vec
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

N = 25

def get_titles(idxs):
    tl = []
    al = []
    
    for i in idxs:
        sha = F_TO_HASH(JSON_FILES[int(i)])
        df_idx = df.index[df['sha'] == sha]
        
        # Sometimes not in the data for whatever reason
        if df_idx.empty:
            tl.append("UNK")
            al.append('')
        else:
            tl.append(df['title'][df_idx[0]])
            abstract = df['abstract'][df_idx[0]]
            
            # Check for NaNs
            if abstract == abstract:
                al.append(abstract[:256])
            else:
                al.append('')
    
    return tl, al
        

def get_labels(fname):
    #print('Loading Embeddings')
    model = Word2Vec.load(fname)
    v = model.wv.vectors

    titles, abstracts = get_titles(model.wv.index2word)
    y = AgglomerativeClustering(n_clusters=N).fit(v).labels_
    
    simplest = TSNE(n_components=2, perplexity=40, n_iter=1500, learning_rate=300)
    v = simplest.fit_transform(v)
    
    return v,y,titles,abstracts

X,y,titles,abstracts = get_labels(NODE_EMBEDDINGS)
print('Embeddings ready')


# Finally, we're done. We build a graph using the method used by [COVID-19 Literature Clustering](https://www.kaggle.com/maksimeren/covid-19-literature-clustering). Though the techniques to generate these graphs are very different, the clusters look largely the same

# In[ ]:


from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.palettes import Category20
from bokeh.transform import linear_cmap
from bokeh.io import show
from bokeh.transform import transform
from bokeh.io import output_notebook
from bokeh.plotting import figure

output_notebook()

# data sources
source = ColumnDataSource(data=dict(
    x= X[:,0], 
    y= X[:,1],
    desc= y, 
    titles= titles,
    abstracts = abstracts
    ))

hover = HoverTool(tooltips=[
    ("Title", "@titles{safe}"),
    ("Abstract", "@abstracts{safe}"),], 
    point_policy="follow_mouse")

mapper = linear_cmap(field_name='desc', 
                     palette=Category20[20],
                     low=min(y) ,high=max(y))

p = figure(plot_width=800, plot_height=800, 
           tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset'], 
           title="t-SNE Covid-19 Articles, Clustered(node2vec), Tf-idf with Plain Text", 
           toolbar_location="right")

p.scatter('x', 'y', size=5, 
          source=source,
          fill_color=mapper,
          line_alpha=0.3,
          line_color="black"
)

show(p)

