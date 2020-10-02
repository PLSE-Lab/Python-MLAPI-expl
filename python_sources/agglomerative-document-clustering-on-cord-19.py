#!/usr/bin/env python
# coding: utf-8

# # A hierarchical agglomerative clustering approach for CORD-19 article categorization
# 
# Many other kernels published in the last few days have focused on LDA for topic detection and made significant advances in that area. This is a method that presents easily recoverable clusters and an intuitive visualization. Unsupervised clustering techniques have been used effectively to categorize and/or classify other data.
# 
# We choose a hierarchical approach because documents are non-lattice, non-real valued data that do not live in Euclidean space. Conventional clustering methods would require some Euclidean embedding for documents, which would require even more validation. Furthermore, this method accounts for the overlapping/concentric structure of topic clusters in document-based data. In this notebook, we extract TF-IDF features to get a representation of the document at a high level instead, and then apply a hierarchical clustering algorithm using the Ward objective function.
# 
# 

# # Setup and Utility functions

# In[ ]:


# for clustering
from scipy.cluster.hierarchy import ward, dendrogram, fcluster, single, complete
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score

# feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer 

# data
import pandas as pd
import numpy as np
import os
import json

# viz
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# other
from tqdm import tqdm
from copy import deepcopy
import time

INPUT_DIR = '/kaggle/input/CORD-19-research-challenge/'
MAX_TITLE_LEN = 70

nltk.download('wordnet')


# In[ ]:


# from xhlulu's kernel

def format_name(author):
    middle_name = " ".join(author['middle'])
    
    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])


def format_affiliation(affiliation):
    text = []
    location = affiliation.get('location')
    if location:
        text.extend(list(affiliation['location'].values()))
    
    institution = affiliation.get('institution')
    if institution:
        text = [institution] + text
    return ", ".join(text)

def format_authors(authors, with_affiliation=False):
    name_ls = []
    
    for author in authors:
        name = format_name(author)
        if with_affiliation:
            affiliation = format_affiliation(author['affiliation'])
            if affiliation:
                name_ls.append(f"{name} ({affiliation})")
            else:
                name_ls.append(name)
        else:
            name_ls.append(name)
    
    return ", ".join(name_ls)

def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    
    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"
    
    return body

def format_bib(bibs):
    if type(bibs) == dict:
        bibs = list(bibs.values())
    bibs = deepcopy(bibs)
    formatted = []
    
    for bib in bibs:
        bib['authors'] = format_authors(
            bib['authors'], 
            with_affiliation=False
        )
        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]
        formatted.append(", ".join(formatted_ls))

    return "; ".join(formatted)

def load_files(dirname):
    filenames = os.listdir(dirname)
    raw_files = []

    for filename in tqdm(filenames):
        filename = dirname + filename
        file = json.load(open(filename, 'rb'))
        raw_files.append(file)
    
    return raw_files

def generate_clean_df(all_files):
    cleaned_files = []
    
    for file in tqdm(all_files):
        features = [
            file['paper_id'],
            file['metadata']['title'],
            format_authors(file['metadata']['authors']),
            format_authors(file['metadata']['authors'], 
                           with_affiliation=True),
            format_body(file['abstract']),
            format_body(file['body_text']),
            format_bib(file['bib_entries']),
            file['metadata']['authors'],
            file['bib_entries']
        ]

        cleaned_files.append(features)

    col_names = ['paper_id', 'title', 'authors',
                 'affiliations', 'abstract', 'text', 
                 'bibliography','raw_authors','raw_bibliography']

    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    clean_df.head()
    
    return clean_df


# # Loading Data
# 
# We load data on the selected portion of the dataset.

# In[ ]:


data_path = INPUT_DIR + "2020-03-13/pmc_custom_license/pmc_custom_license/"
files = load_files(data_path)
print("Loaded {} files".format(len(files)))
df = generate_clean_df(files)
df.head()


# # Clustering
# 
# Distance metric: Cosine distance, using (1,3)-shingles generated via TF-IDF feature extraction. Shingles are generated in lowercase, with options to stem or not stem the tokens.

# In[ ]:


stemmer = nltk.stem.snowball.SnowballStemmer("english")
lemmatizer = WordNetLemmatizer() 
translator = str.maketrans('', '', "!?;(),[]")

def tokenize_and_lemmatize_and_stem(text):
    """
        @param text: document text. Can be a synopsis, full text, or any multi-sentence text.
        
        This returns a list of lemmatized, stemmed, lowercase (if applicable) word-level tokens given a 
        multi-sentence text.
    """
    text = text.translate(translator)
    tokens = [stemmer.stem(lemmatizer.lemmatize(word.lower())) for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    return tokens

def tokenize(text):
    """
        @param text: document text. Can be a synopsis, full text, or any multi-sentence text.
        
        This returns a list of lowercase (if applicable) word-level tokens given a 
        multi-sentence text.
    """
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    return tokens


# We use a TF-IDF Vectorizer to generate (1,3)-shingles and create a TF-IDF matrix. The current threshold includes only terms with document frequencies in the interval (0.005,0.8), though this can be tuned. The idea is that if a term has document frequency outside that interval, it is likely to be extremely obscure and therefore irrelevant, or ubiquitous and also irrelevant (think stopwords). We use the sublinear tf scaling technique to "normalize" term frequencies on a log-scale. Using the TF-IDF matrix, we then use the cosine distance metric (1 - cosine similarity) to create a sparse pairwise distance matrix. We pass this into the hierarchical clustering algorithm to receive a linkage matrix for plotting.

# ## Hierarchical clustering
# 
# We experiment with multiple hierarchical clustering criteria:
# * Ward (maximize the decrease in intra-cluster variance with respect to all clusters)
# * Single (move point to cluster of nearest neighbor)
# * Complete (move point to the cluster with minimum distance to farthest point)
# 
# Ward clustering yielded reasonable results, as did complete clustering. Many cases Using the single criterion results in a degenerate case.

# In[ ]:


tfidf_vectorizer = TfidfVectorizer(ngram_range=(3,4), max_df=0.9, min_df=0.005, sublinear_tf=True, tokenizer=tokenize_and_lemmatize_and_stem)
data = df

tfidf_matrix = tfidf_vectorizer.fit_transform(data.text)
dist = 1 - cosine_similarity(tfidf_matrix)
dist = dist - dist.min() # get rid of some pesky floating point errors that give neg. distance
linkage_matrix = ward(dist) # replace with complete, single, or other scipy.cluster.hierarchical algorithms


# ## Visualization
# 
# Dendrogram visualization. We visualize the titles of each paper next to the corresponding leaf to qualitatively evaluate the clustering.
# 
# Changing the `color_threshold` kwarg in the call to `dendrogram()` will affect how the clusters are displayed by increasing or decreasing the 
# minimum distance necessary for `dendrogram()` to color a cluster differently.

# In[ ]:


MAX_COPHENETIC_DIST = max(linkage_matrix[:,2]) * 0.39 # max distance between points to be considered together. can be tuned.

fig, ax = plt.subplots(figsize=(15, 80)) # set size
ax = dendrogram(linkage_matrix, orientation="right", color_threshold=MAX_COPHENETIC_DIST, leaf_font_size=4,
                labels=data.title.apply(lambda x: x if len(x) < MAX_TITLE_LEN else x[:MAX_TITLE_LEN  - 3] + "...").tolist())

plt.tick_params(axis= 'x', which='both',  bottom='off', top='off',labelbottom='off')

plt.tight_layout() #show plot with tight layout
plt.savefig('ward_clusters_all.png', dpi=300)


# # Analysis
# 
# We now take the raw clusters achieved and use the silhouette score and elbow method to evaluate.

# In[ ]:


def silhouette_k(distance_matrix, linkage_matrix, max_k=20):
    scores = []
    for i in range(2, max_k+1):
        clusters = fcluster(linkage_matrix, i, criterion='maxclust')
        score = silhouette_score(distance_matrix, clusters, metric='precomputed')
        print("Silhouette score with {} clusters:".format(i), score)
        scores.append(score)
    plt.title("Silhouette score vs. number of clusters")
    plt.xlabel("# of clusters")
    plt.ylabel("Score (higher is better)")
    plt.plot(np.arange(2, max_k+1), scores)
    plt.show()
    return scores
    


# In[ ]:


_ = silhouette_k(dist, linkage_matrix)


# Silhouette score sharply drops off after k=5 on complete clustering and k=10 on Ward clustering. I ultimately opted for Ward clustering because of its intra-cluster variance minimization objective. I then retuned the `MAX_COPHENETIC_DIST` parameter such that the visualization would show 10 clusters. A qualitative high level description of the clusters from top to bottom follows. Please feel free to examine the image yourself.
# 
# * **Cluster 1: Zoonotic diseases (magenta).** Mostly concerns diseases originally found in other animals.
# * **Cluster 2: Respiratory viruses, e.g. bocavirus, rhinovirus (cyan).** Concerns viruses that affect the respiratory tract.
# * **Cluster 3: MERS (red).** Self-explanatory.
# * **Cluster 4: indetereminate (green).** Unable to determine a topic based on this cluster.
# * **Cluster 5: indetereminate (black).**
# * **Cluster 6: policy/institutional considerations (yellow).** Mostly about preparedness, policy evaluation, etc.
# * **Cluster 7: Treatments (magenta).**
# * **Cluster 8: Treatments (cyan).**
# * **Cluster 9: Indeterminate, possibly cell biology-related (red).** Has a lot of cell biology-sounding words that I don't know. I'm unable to describe this cluster qualitatively without domain knoweldge.
# * **Cluster 10: DNA-related (green).** Mostly pertains to DNA transcription, sequencing, and related topics.

# # Limitations/Assumptions
# 
# While agglomerative document clustering achieves a somewhat interpretible clustering, and can also partition data into a number of clusters, there are a few key limitations to consider.
# 
# * **The how-many-clusters problem.** This mode of clustering gives little insight into whether the clusters discovered actually constitute semantically meaningul topic differences. That is; it is unclear 1) how a human would divide this dataset into topics, and 2) if our discovered  clusters map onto human "concepts" well.
# * **Single, hard assignment.** As opposed to other EM-style distribution estimation algorithms or LDA, which yield a "soft" assignment by assigning a distribution over a set of latent variables, this method only assigns each document unequivocally to a single cluster.
# * **Distance metric.** Tf-idf has been shown to be successful on various tasks. However, the distance metric here is the cosine-distance between the tf-idf vectors of two documents. Let's call shingles (n-grams)  with high tf-idf scores "important." Here I am assuming 1) that tf-idf ideally captures "important" n-grams in the corpus, and that 2) two closely related articles will have similarly co-occuring "important" n-grams.
# * **Space.** Calculating a pairwise distance matrix will invariably take O(n^2) memory. This will likely not scale as the set of documents grows.
# * **Feature Extraction.** This is very close to a CBOW (continuous bag of words) style model. The document-based representations here account for very little context (accounted for in the n-gram feature extraction only). To see why this might be a problem, one paper might use a particular set of words in an endorsing context; another paper might use that same turn of phrase in a critical context (e.g. counterargument (?), related work, etc.). This is left out of our feature extraction. 

# # Future Work
# 
# A list of future tasks to improve on this work:
# * Tuning parameters (shingle size, TF-IDF thresholds, etc.)
# * Alternative distance metrics for alternative clustering methods (shingles->Jaccard sim., embedding->Euclidean clustering)
# * Alternative features (text vs. abstracts-only, etc.)
# * Cluster quality evaluation (both with and without ground-truth categories)
# 
# I am most interested in exploring cluster quality evaluation at this time. Specifically, I am exploring ways to address the limitations above with and without expert knowledge, since I lack the domain knowledge to meaningfully label the articles myself.

# # Acknowledgements
# 
# Special thanks to xhulu for their preprocessing code, and Brandon Rose for [this excellent resource on implementing hierarchical document clustering with sklearn](http://brandonrose.org/clustering#Hierarchical-document-clustering). Further analysis of clustering algorithms in Mining of Massive Datasets (Ullman, Rajaravan, and Leskovec 2010) assisted with the creation of this as well.
