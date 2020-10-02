#!/usr/bin/env python
# coding: utf-8

# ## Table of Content <a id="toc"></a>
# * [Global Variables](#gv)
# * [1. Data Preprocessing](#data_preprocessing)
#     * [1.1 Importing Data and Separating Data of Our Interest](#1.1)
#     * [1.2 Creating Preprocessing Function and Applying it on Our Data](#1.2)
#     * [1.3 Creating TF-IDF Matrix](#1.3)
# * [2. Apply SVD to TF-IDF Matrix](#apply_svd)
#     * [2.1 Create Term and Document Representation](#2.1)
#     * [2.2 Visulize Those Representation](#2.2)
# * [3 Information Retreival Using LSA](#ir_lsa)
# * [4 References](#references)

# In[ ]:


# Global Variables 
K = 2 # number of components
query = 'nice good price'


# ##  1. Data Preprocessing <a id="data_preprocessing"></a>

# ### 1.1 Importing Data and Separating Data of Our Interest <a id="1.1"></a>

# In[ ]:


import pandas as pd
import numpy as np

# Data filename
dataset_filename = "../input/Womens Clothing E-Commerce Reviews.csv"

# Loading dataset
data = pd.read_csv(dataset_filename, index_col=0)

# We are reducing the size of our dataset to decrease the running time of code
datax = data.loc[data['Clothing ID'] == 1078 , :]


# Delete missing observations for variables that we will be working with
for x in ["Recommended IND","Review Text"]:
    datax = datax[datax[x].notnull()]

# Keeping only those features that we will explore
datax = datax[["Recommended IND","Review Text"]]

# Resetting the index
datax.index = pd.Series(list(range(datax.shape[0])))
    
print('Shape : ',datax.shape)
datax.head()


# ### 1.2 Creating Preprocessing Function and Applying it on Our Data <a id="1.2"></a>

# In[ ]:


from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

wordnet_lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'[a-z]+')
stop_words = set(stopwords.words('english'))

def preprocess(document):
    document = document.lower() # Convert to lowercase
    words = tokenizer.tokenize(document) # Tokenize
    words = [w for w in words if not w in stop_words] # Removing stopwords
    # Lemmatizing
    for pos in [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]:
        words = [wordnet_lemmatizer.lemmatize(x, pos) for x in words]
    return " ".join(words)


# In[ ]:


datax['Processed Review'] = datax['Review Text'].apply(preprocess)

datax.head()


# ### 1.3 Creating TF-IDF Matrix <a id="1.3"></a>

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
TF_IDF_matrix = vectorizer.fit_transform(datax['Processed Review'])
TF_IDF_matrix = TF_IDF_matrix.T

print('Vocabulary Size : ', len(vectorizer.get_feature_names()))
print('Shape of Matrix : ', TF_IDF_matrix.shape)


# ## 2. Apply SVD to TF-IDF Matrix <a id="apply_svd"></a>

# ### 2.1 Create Term and Document Representation  <a id="2.1"></a>

# In[ ]:


import numpy as np

# Applying SVD
U, s, VT = np.linalg.svd(TF_IDF_matrix.toarray()) # .T is used to take transpose and .toarray() is used to convert sparse matrix to normal matrix

TF_IDF_matrix_reduced = np.dot(U[:,:K], np.dot(np.diag(s[:K]), VT[:K, :]))

# Getting document and term representation
terms_rep = np.dot(U[:,:K], np.diag(s[:K])) # M X K matrix where M = Vocabulary Size and N = Number of documents
docs_rep = np.dot(np.diag(s[:K]), VT[:K, :]).T # N x K matrix 


# ### 2.2 Visulize Those Representation <a id="2.2"></a>

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.scatter(docs_rep[:,0], docs_rep[:,1], c=datax['Recommended IND'])
plt.title("Document Representation")
plt.show()


# In[ ]:


plt.scatter(terms_rep[:,0], terms_rep[:,1])
plt.title("Term Representation")
plt.show()


# ## 3 Information Retreival Using LSA <a id="ir_lsa"></a>

# In[ ]:


# This is a function to generate query_rep

def lsa_query_rep(query):
    query_rep = [vectorizer.vocabulary_[x] for x in preprocess(query).split()]
    query_rep = np.mean(terms_rep[query_rep],axis=0)
    return query_rep


# In[ ]:


from scipy.spatial.distance import cosine

query_rep = lsa_query_rep(query)

query_doc_cos_dist = [cosine(query_rep, doc_rep) for doc_rep in docs_rep]
query_doc_sort_index = np.argsort(np.array(query_doc_cos_dist))

print_count = 0
for rank, sort_index in enumerate(query_doc_sort_index):
    print ('Rank : ', rank, ' Consine : ', 1 - query_doc_cos_dist[sort_index],' Review : ', datax['Review Text'][sort_index])
    if print_count == 4 :
        break
    else:
        print_count += 1


# ##  4. References <a id="references"></a>
# * [Latent Semantic Analysis (Tutorial)](https://www.engr.uvic.ca/~seng474/svd.pdf)
