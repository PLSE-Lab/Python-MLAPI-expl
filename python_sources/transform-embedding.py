#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(train_df.question_text.values)

tfidf=vectorizer.fit_transform(train_df.question_text.values)
woord=vectorizer.get_feature_names()

vectorizer.fit_transform(train_df[train_df.target==0].question_text.values)
woord0=vectorizer.get_feature_names()

vectorizer.fit_transform(train_df[train_df.target==1].question_text.values)
woord1=vectorizer.get_feature_names()

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 
woord01=intersection(set(woord0),set(woord1))
print(len(woord01),len(woord1),len(woord0))
def exclinters(lst1, lst2): 
    lst3 = [value for value in lst1 if value not in lst2] 
    return lst3 
wordsimpf=exclinters(set(woord1),set(woord01))
print(len(woord),len(wordsimpf))


# In[ ]:


EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE)  )


# In[ ]:


glove=pd.DataFrame([] )
for xi in range(int(len(woord)/1000)+1):
    
    emb =pd.DataFrame([] )
    for word in woord[xi*1000:xi*1000+1000]:
        emb=emb.append(pd.DataFrame(embedding_index.get(word),columns=[word] ).T)
    glove=glove.append(emb)    
                    
        #print(emb.shape)
    


# In[ ]:


EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)


# In[ ]:


wiki=pd.DataFrame([] )
for xi in range(int(len(woord)/1000)+1):
    
    emb =pd.DataFrame([] )
    for word in woord[xi*1000:xi*1000+1000]:
        emb=emb.append(pd.DataFrame(embedding_index.get(word),columns=[word] ).T)
    wiki=wiki.append(emb)    
                    
        #print(emb.shape)


# In[ ]:


wiki


# In[ ]:


del EMBEDDING_FILE
del embedding_index


# In[ ]:


wiki=wiki.dropna()
glove=glove.dropna()


# In[ ]:





# In[ ]:


# from https://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
def normalized(a, axis=-1, order=2):
    """Utility function to normalize the rows of a numpy array."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def make_training_matrices(source_dictionary, target_dictionary, bilingual_dictionary):
    """
    Source and target dictionaries are the FastVector objects of
    source/target languages. bilingual_dictionary is a list of 
    translation pair tuples [(source_word, target_word), ...].
    """
    source_matrix = pd.DataFrame([])
    target_matrix = pd.DataFrame([])

    for (source, target) in bilingual_dictionary:
        if source in source_dictionary and target in target_dictionary:
            source_matrix.append(source_dictionary.loc[source])
            target_matrix.append(target_dictionary.loc[target])

    # return training matrices
    return source_matrix, target_matrix

def learn_transformation(source_matrix, target_matrix, normalize_vectors=True):
    """
    Source and target matrices are numpy arrays, shape
    (dictionary_length, embedding_dimension). These contain paired
    word vectors from the bilingual dictionary.
    """
    # optionally normalize the training vectors
    if normalize_vectors:
        source_matrix = normalized(source_matrix)
        target_matrix = normalized(target_matrix)

    # perform the SVD
    product = np.matmul(source_matrix.transpose(), target_matrix)
    print(product)
    U, s, V = np.linalg.svd(product)

    # return orthogonal transformation which aligns source language to the target
    return np.matmul(U, V)


# In[ ]:



overlap = list(wiki.index & glove.index)
bilingual_dictionary = [(entry, entry) for entry in overlap]

# form the training matrices
source_matrix=wiki.loc[overlap]
target_matrix=glove.loc[overlap]
print( source_matrix.shape,target_matrix.shape )

# learn and apply the transformation
transform = learn_transformation(source_matrix.values, target_matrix.values)
uniform=np.dot( wiki.values,transform )
uniform


# In[ ]:


uniform=pd.DataFrame(uniform,index=wiki.index)
uniform.shape


# In[ ]:


from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity
print( cosine_similarity(wiki.loc[['cat','dog']],wiki.loc[['cat','dog']]) )
print( cosine_similarity(wiki.loc[['cat','dog']],glove.loc[['cat','dog']]) )

#the uniform matrix is the wiki database transformed to the glove
print( cosine_similarity(uniform.loc[['cat','dog']],glove.loc[['cat','dog']]) )


# In[ ]:


# form the training matrices
source_matrix=glove.loc[overlap]
target_matrix=wiki.loc[overlap]
print( source_matrix.shape,target_matrix.shape )

# learn and apply the transformation
transform = learn_transformation(source_matrix.values, target_matrix.values)
uniform=np.dot( glove.values,transform )
uniform


# In[ ]:


uniform=pd.DataFrame(uniform,index=glove.index)
uniform.shape


# In[ ]:


from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity
print( cosine_similarity(wiki.loc[['cat','dog']],wiki.loc[['cat','dog']]) )
print( cosine_similarity(wiki.loc[['cat','dog']],glove.loc[['cat','dog']]) )

#the uniform matrix is the gloe database transformed to the wiki
print( cosine_similarity(uniform.loc[['cat','dog']],wiki.loc[['cat','dog']]) )


# In[ ]:


glove.to_csv('glove.csv')
wiki.to_csv('wiki.csv')
uniform.to_csv('uniform.csv')

