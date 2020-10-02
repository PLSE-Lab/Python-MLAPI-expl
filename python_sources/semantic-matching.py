#!/usr/bin/env python
# coding: utf-8

# # Semantic Matching:
# # Word Embedding for Prioritizing Research Papers
# 
# ![CORD19](https://pages.semanticscholar.org/hs-fs/hubfs/covid-image.png?width=300&name=covid-image.png)
# 
# ## Introduction
# COVID-19 Open Research Dataset (CORD-19) is a free dataset of academic papers, aggregated by a coalition of leading research groups, about COVID-19 and the coronavirus family of viruses. The dataset can be found on Semantic Scholar and there is a research challenge on Kaggle. This dataset is being given to the global research community to apply recent developments in natural language processing and other AI techniques to generate new insights in support of the ongoing battle against this infectious disease. These approaches are becoming increasingly urgent due to the rapid acceleration in the new literature on coronavirus, which makes it difficult for medical research community to keep up. In this notebook we try to prioterize research papers based on semantic search instead of classical search. methods
# 
# ## Approach
# In this notebook, we use gensim's word2vec in order to generate word embeddings for the research papers' abstracts to use as our corpus. 
# 
# #### Semantic Matching
# Word embeddings represent words as d-dimensional dense vectors. The similarity or the distance between the vectors of words in the embedding space measure the relatedness between them. 
# 
# #### IWCS 
# The IDF re-weighted word centroid similarity (IWCS) model used word embeddings to construct a d-dimensional vector representing a passage (abstract in our case). In this model, the word vectors of the given text are aggregated into a single vector using a linear weighted combination of its word vectors.
# The centroid vector of the query can also be computed in the same manner. Finally, to rank a text according to a query we use the cosine distnace between them.
# 
# Inspired from: [The Semantic Web: 16th International Conference](https://books.google.com.eg/books?id=PxaaDwAAQBAJ&dq=The+Semantic+Web:+16th+International+Conference,+ESWC&source=gbs_navlinks_s)
# 
# 

# # Installing/Loading packagaes

# In[ ]:


from IPython.utils import io
with io.capture_output() as captured:
    get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz')
import spacy
import string
import warnings

import numpy as np
import pandas as pd

from pprint import pprint
from IPython.utils import io
from tqdm.notebook import tqdm
from gensim.models import Word2Vec
from IPython.core.display import HTML, display
from spacy.lang.en.stop_words import STOP_WORDS

warnings.filterwarnings('ignore')


# # Data loading and preprocessing
# We consider the paper abstract only, but the approach could also be applied to the whole text body.
# 
# 

# In[ ]:


root_path = '/kaggle/input/CORD-19-research-challenge'
metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str,
    'abstract': str
})
meta_df.head()


# In[ ]:


df_cord = pd.DataFrame(columns=['paper_id', 'title','abstract', 'doi'])
df_cord['paper_id'] = meta_df.sha
df_cord['title'] = meta_df.title
df_cord['abstract'] = meta_df.abstract
df_cord['doi'] = meta_df.doi

df_cord.head()


# In[ ]:


df_cord.info()


# Dropping null and duplicate values.

# In[ ]:


df_cord.drop_duplicates(['abstract'], inplace=True)
df_cord.dropna(inplace=True)
df_cord.info()


# ## Text Tokenizing
# For preprocessing we use scispaCy, which is a Python package containing spaCy models for processing biomedical, scientific or clinical text.

# In[ ]:


import en_core_sci_lg
nlp = en_core_sci_lg.load(disable=["tagger", "ner"])# disabling Named Entity Recognition for speed
nlp.max_length = 3000000
def spacy_tokenizer(sentence):
    return ' '.join([word.lemma_ for word in nlp(sentence) if not (word.like_num or word.is_stop or word.is_punct or word.is_space or len(word)==1)])


# In[ ]:


customized_stop_words = [
    'rights', 'reserved', 'permission', 'use', 'used', 'using', 'biorxiv', 'medrxiv', 'license',
    'doi', 'preprint', 'copyright', 'org', 'https', 'et', 'al', 'author', 'figure', 'table',
     'fig', 'fig.', 'al.', 'Elsevier', 'PMC', 'CZI',
    '-PRON-', 'usually'
]
punctuations = string.punctuation
stopwords = list(STOP_WORDS)
# Mark them as stop words
stopwords.extend(customized_stop_words)


# ## Applying the tokenizer on the abstarcts and creating a new column `tokenized_abstract`

# In[ ]:


tqdm.pandas()
df_cord["tokenized_abstract"] = df_cord["abstract"].progress_apply(spacy_tokenizer)
df_cord.head()


# ## Corpus Sentencization
# We apply spacy's sentencizer to split all the abstracts into separate sentences so we can use them as our word2vec corpus.

# In[ ]:


abstracts = df_cord['abstract'].values

nlp.add_pipe(nlp.create_pipe('sentencizer'), before="parser")
word2vec_corpus = []

for i in tqdm(range(0, len(abstracts))):
    doc = nlp(abstracts[i])
    word2vec_corpus.extend([spacy_tokenizer(sentence.string.strip()).split(" ") for sentence in doc.sents])


# In[ ]:


word2vec_corpus[:5]


# # Training the Model (word2vec)
# We will use gensim's word2vec and train it on the processed abstract.
# 
# ## Important parameters:
# * `min_count` Ignores all words with total absolute frequency lower than this - (2, 100)
# * `window` The maximum distance between the current and predicted word within a sentence. E.g. window words on the left and window words on the left of our target - (2, 10)
# * `size` Dimensionality of the feature vectors. - (50, 300)
# * `workers` = Use these many worker threads to train the model (=faster training with multicore machines)
# * `sg`enable skipgram model

# In[ ]:


# Train the gensim word2vec model with our corpus
import multiprocessing
cores = multiprocessing.cpu_count() # Count the number of cores in a computer
size = 50
model = Word2Vec(word2vec_corpus, min_count=5,size= size,workers=cores-1, window =5, sg = 1)


# # Text Centroid Calculation
# we will calculate the centroid for each abstract using the vectors of all the words in the abstract.

# In[ ]:


df_cord["centroid"] = [[0.0]*size]*df_cord.shape[0]
for index, row in df_cord.iterrows():
    abstract = row['tokenized_abstract']
    centroid = np.array([0.0]*size)
    for word in abstract.split(" "):
        try:
            word_vector = model[word]
        except:
            continue
        centroid = np.add(centroid, word_vector)

    df_cord.at[index,'centroid'] = centroid.tolist()

df_cord.head()


# # Ranking research papers
# Given a query, we compute its centroid and then determine the top `k` semantically similar papers to the query according to the cosine similarity score.

# In[ ]:


def get_top_k_docs(model, query, df_cord, k) :
    cosine_distance = []
    
    vectorized_query = []
    for word in spacy_tokenizer(query).split(" "):
        try:
            vectorized_query.append(model[word])
        except:
            continue
    
    for _, row in df_cord.iterrows():
        centroid = row['centroid']
        total_simalirity = 0
        for word_vec in vectorized_query:
            word_simalirity = np.dot(word_vec, centroid)/(np.linalg.norm(word_vec)*np.linalg.norm(centroid))
            total_simalirity += word_simalirity
        cosine_distance.append((row['title'], row['doi'],row['abstract'], total_simalirity)) 
    
    
    cosine_distance.sort(key=lambda x:x[3], reverse=True) #Sort according to cosine simalirity in descending order
    return cosine_distance[:k]


# In[ ]:


get_top_k_docs(model=model,query='origin of coronavirus',df_cord=df_cord,k=10)


# In[ ]:


def search(search_query,n_docs=5):
    html = """
        <html>
            <body>
                <ol>
            """
    results = get_top_k_docs(model=model,query=search_query,df_cord=df_cord,k=n_docs)
    for result in results:
        paper_name = result[0]
        paper_doi = result[1]
        paper_abstract = result[2]
        paper_link = "https://doi.org/" + str(paper_doi)
        html += f"""            
                <li id="result-1">
                    <article>
                        <header>
                            <a href="{paper_link}">
                                <h2>{paper_name}</h2>
                            </a>
                        </header>
                        <p>{paper_abstract}</p>
                    </article>
                  </li>
                """
    html += "</body></html>"
    display(HTML(html))


# # What is known about transmission, incubation, and environmental stability?

# In[ ]:


search('transmission and incubation of coronavirus')


# # What do we know about COVID-19 risk factors?
# 

# In[ ]:


search('coronavirus risk factors')


# # What do we know about virus genetics, origin, and evolution?
# 

# In[ ]:


search('genetics origin and evolution of coronavirus')


# 
# # What do we know about vaccines and therapeutics?
# 

# In[ ]:


search('coronavirus vaccines and therapeutics')


# # What do we know about non-pharmaceutical interventions?
# 

# In[ ]:


search('non-pharamceutical interventions of coronavirus')


# # What has been published about ethical and social science considerations?
# 

# In[ ]:


search('ethical and social science considerations of coronavirus')


# # What has been published about medical care?
# 

# In[ ]:


search('coronavirus medical care')


# # What do we know about diagnostics and surveillance?
# 

# In[ ]:


search('coronavirus diagnostics and surveillance')


# # What has been published about information sharing and inter-sectoral collaboration?

# In[ ]:


search('coronavirus information sharing and colaboration')


# In[ ]:




