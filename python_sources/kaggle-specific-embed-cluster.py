# %% [code] {"_kg_hide-input":true,"_kg_hide-output":true}
## Imports (code & data)
from math import sqrt, floor
import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from nltk.tokenize import RegexpTokenizer

# read in pre-tuned vectors
# make sure to add this kernel to your notebook: 
# https://www.kaggle.com/rebeccaturner/fine-tuning-word2vec-2-0
VECTORS = pd.read_csv("../input/fine-tuning-word2vec-2-0/kaggle_word2vec.model", 
                      delim_whitespace=True,
                      skiprows=[0],
                      header=None
                     )
# set words as index rather than first column
VECTORS.index = VECTORS[0]
VECTORS.drop(0, axis=1, inplace=True)

# get vectors for each word in post
# TODO: can we vectorize this?
def vectors_from_post(post):
    """Get word vectors for all words in text"""
    all_words = []

    for words in post:
        all_words.append(words)

    return(VECTORS[VECTORS.index.isin(all_words)])

# create document embeddings from post
def doc_embed_from_post(post):
    """Get document embeddings from post embeddings
    TODO: something fancier than mean"""
    test_vectors = vectors_from_post(post)
    return test_vectors.mean()

def tokenize(sentences):
    """Regex tokenizer (won't work for Thai, CJK, etc.)"""
    tokenizer = RegexpTokenizer(r'\w+')
    sample_data_tokenized = [w.lower() for w in sentences]
    sample_data_tokenized = [tokenizer.tokenize(i) for i in sample_data_tokenized]
    
    return sample_data_tokenized

def get_num_of_posts(sample_posts):
    """ number of posts in dataframe"""
    num_of_posts = sample_posts.shape[0]
    
    return num_of_posts 

def get_num_of_clusters(sample_posts):
    "get number of clusters (sqrt of # of posts)"
    num_of_posts = get_num_of_posts(sample_posts)
    
    # Number of clusters is square root of the # of posts (rounded down)
    number_clusters = floor(sqrt(num_of_posts))
    
    return number_clusters 

def get_keyword_sets(sample_posts):
    """tokenize & get word sets for input texts """ 
    keywords_tokenized = tokenize(sample_posts)
    keyword_sets = [set(post) for post in keywords_tokenized]
    
    return keyword_sets

def get_embeddings_for_posts(sample_posts, vector_dim=300):
    """ Given texts, return embedding for each text.
    Assumes vectors of dim 300.
    """
    keyword_sets = get_keyword_sets(sample_posts)
    num_of_posts = get_num_of_posts(sample_posts)
    
    # create empty array for document embeddings
    doc_embeddings = np.zeros([num_of_posts, vector_dim])

    # get document embeddings for posts
    for i in range(num_of_posts):
        embeddings = np.array(doc_embed_from_post(keyword_sets[i]))
        if np.isnan(embeddings).any():
            doc_embeddings[i, :] = np.zeros([1, vector_dim])
        else:
            doc_embeddings[i, :] = embeddings
            
    return doc_embeddings


def get_spectral_clusters(sample_posts):
    """Return spectral clusters given texts as input"""
    number_clusters = get_num_of_clusters(sample_posts)
    doc_embeddings = get_embeddings_for_posts(sample_posts)
    
    clustering = SpectralClustering(n_clusters=number_clusters,
                                    assign_labels="discretize",
                                    n_neighbors=number_clusters).fit(doc_embeddings)
    return clustering
