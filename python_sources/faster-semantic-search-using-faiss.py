#!/usr/bin/env python
# coding: utf-8

# # **Faster Semantic Search over BERT embeddings using Faiss**
# Recently I was experimenting with COVID-19 dataset using BERT. I generated BERT embeddings and later used those embeddings for semantic search. I found out that the semantic search using traditional methods were very slow , especially for larger datasets like COVID-19 Open-Research data which has 44k+ papers. Here is a method to fasten up semantic search process using FAISS. 
# 
# In this notebook, I will create embeddings from scratch and later build-up an index using FAISS.
# 
# 1, Install the requirements

# In[ ]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install faiss-gpu')


# We will use Sci-BERT to generate embeddings over COVID-19 paper titles, we can also build embeddings for abstract and contents. For this demonstration we will consider only titles of the articles, we will encode all the titles. (All 44k). First, download Sci-BERT model.

# In[ ]:


get_ipython().system('wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_uncased.tar -O scibert.tar')
get_ipython().system(' tar -xvf scibert.tar')


# In[ ]:


get_ipython().system('pip install sentence-transformers')


#  Now, we import all the libraries and create a transformer pytorch model. We will load the model on GPU. You can remove `sciBert.cuda()` in case you don't want to use a GPU. 

# In[ ]:


import torch
import transformers
import numpy as np 
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

#globals 
MODEL = 'scibert_scivocab_uncased'

#load the model
sciBert = transformers.BertModel.from_pretrained(MODEL)

#create a transformer tokenizer for BERT
tokenizer = transformers.BertTokenizer.from_pretrained(MODEL, do_lower_case=True)

print(type(sciBert))

sciBert.eval()
sciBert.cuda(0)


# Next we define some important functions that will help us in generating the embeddings. These functions will be used throughout the notebook. `Use embedding_fn_cpu` if not using GPU.

# In[ ]:


def embedding_fn(model, text) :

  if not isinstance(model, transformers.modeling_bert.BertModel) :
    print('Model must be of type transformers.modeling_bert.BertModel, but got ', type(model))
    return

  with torch.no_grad():
    #generate tokens :
    tokens = tokenizer.encode(text)
    #expand dims : 
    batch_tokens = np.expand_dims(tokens, axis = 0)
    batch_tokens = torch.tensor(batch_tokens).cuda()
    #print(type(batch_tokens))
    #generate embedding and return hidden_state : 
    return model(batch_tokens)[0].cpu()

def embedding_fn_cpu(model, text) :

  if not isinstance(model, transformers.modeling_bert.BertModel) :
    print('Model must be of type transformers.modeling_bert.BertModel, but got ', type(model))
    return

  with torch.no_grad():
    #generate tokens :
    tokens = tokenizer.encode(text, max_length = 512)
    #expand dims : 
    batch_tokens = np.expand_dims(tokens, axis = 0)
    batch_tokens = torch.tensor(batch_tokens)
    #print(type(batch_tokens))
    #generate embedding and return hidden_state : 
    return model(batch_tokens)[0]

def compute_mean(embedding):

  if not isinstance(embedding, torch.Tensor):
    print('Embedding must be a torch.Tensor')
    return 
  
  return embedding.mean(1)


def compute_cosine_measure(x1, x2):

  #given two points in vector space, measure cosine distance
  return cosine_similarity(x1, x2)


def compute_distance(x1, x2):
  #replace this with your own measure
  return compute_cosine_measure(x1.detach().numpy(), x2.detach().numpy())


# The actual information is in CSV format, present in `metadata.csv`, we will build an index that contains information we need, we can use this index to present the human-readable information. Add the dataset to the notebook if you haven't added yet. It will be loaded in the following path, it is a read-only mount.

# In[ ]:


get_ipython().system('ls /kaggle/input/CORD-19-research-challenge')


# ### Download pre-grenerated embeddings for Titles :
# For easy demo, I have already uploaded pre-generated CORD-19 title embeddings. You can get it from here : [CORD-19 title embeddings](https://www.kaggle.com/narasimha1997/cord-19-title-embeddings)

# In[ ]:


dataset = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')


import json

#We generate Index File and of key-value pair , each dict has 2 values : cord_uid and title. This we save in a separate CSV file
def generate_mapping_index(dataframe):

  index_map = {}

  for index, row in dataframe.iterrows():
    index_map[index] = {
        "cord_uid" : row['cord_uid'],
        "title" : row['title'],
        "abstract" : row['abstract'],
        "url" : row['url']
    }
  
  return index_map


index_map = generate_mapping_index(dataset)
open('index.json', 'w').write(json.dumps(index_map))
dataset.head()


#   We are interested only in `cord_uid` , `title` , `abstract` and `url`, we will present these fileds upon search completion. I have dumped the `index.json` file for later use. I have prepared the code to process data as chunks, If you want to save numpy gz arrays as chunks, you can reduce the CHUNK_SIZE_EACH to a lesser value, The simple logic below will split the dataframe, embed them individually and save them as chunks, now I am creating a single big chunk of 44k records. Embeddings are generated by GPU. Skip this step if you have downloaded pre-generated embeddings.

# In[ ]:


CHUNK_SIZE_EACH = 44000



def __embedding(text):
  return compute_mean(embedding_fn(sciBert, text))



def compute_bert_embeddings(dataframe_chunk, current_index, end_marker):

  np_chunk = __embedding(dataframe_chunk.loc[current_index * end_marker]['title']).detach().numpy()
  #np_chunk = np_chunk.reshape(np_chunk.shape[1])

  for idx in range(1, end_marker):

    try:
      embedding = __embedding(dataframe_chunk.loc[(current_index * end_marker) + idx]['title']).detach().numpy()
      #embedding = embedding.reshape(embedding.shape[1])
      np_chunk = np.append(np_chunk, embedding, axis = 0)
      print('\r {}'.format(np_chunk.shape), end = '')
    except Exception as e:
      print(e)
      np_chunk = np.append(np_chunk, np.zeros(shape = (1, 768)), axis = 0)
      continue 

  print(np_chunk.shape)
  np.savez_compressed('title_{}'.format(current_index), a = np_chunk)


def compute_embeddings_and_save(dataframe):

  n_rows = len(dataframe)
  
  chunk_sizes = n_rows // CHUNK_SIZE_EACH
  remaining = n_rows - chunk_sizes * CHUNK_SIZE_EACH

  for i in range(1):

    compute_bert_embeddings(dataframe[i * CHUNK_SIZE_EACH : (i * CHUNK_SIZE_EACH) + CHUNK_SIZE_EACH ], i, CHUNK_SIZE_EACH)


#Un-comment this if you want to regenerate embeddings.
#compute_embeddings_and_save(dataset)


# This will generate embeddings for COVID-19 article titles, These embeddings can be used for semantic search. In simple words, BERT is like a hash-mapping function. It maps arbitrary length sentences ( presented as tokens ) to fixed-length word-vectors, These word-vectors are n-dimensional numerical values. These word-vectors contain enough information to semantically present the given sentence. How BERT was able to generate this? It was able to do so because it has developed a language model of the data it was trained on. How it developed the language model is quite complex and is outside the scope of the notebook, You can refer to BERT research paper to understand more. The above code will handle NaN values implicitly, so no need of data-cleaning to remove NaNs. The dataset has atleast 100 titles missing from the CSV file, those are ignored. Now, let us verifiy the npz file.

# In[ ]:


get_ipython().system('ls')


# `title_0.npz` is generated. We will load and verifiy its dimension.

# In[ ]:


embeddings = np.load('/kaggle/input/cord-19-title-embeddings/title_0.npz')['a']
embeddings.shape


# `embeddings` is a vector of (40000, 768) dimensions, this is our look-up vector, Now we will examine various ways of performing semantic search.

# **Method -1 : Normal Brute-force search using cosine-distance measure**
# We will do a normal brute-force cosine-distance calculation and display top 20 matches.

# In[ ]:


import time

#print(index_map.keys())
def index_to_title(indexes):
    
    for i, idx in enumerate(indexes) :
        print('{}. {}'.format(i, index_map[idx]['title']))

def do_consine_search(embeddings, query_text, model, top_k):
    
    n_embeddings = embeddings.shape[0]
    
    embedding_q = compute_mean(embedding_fn(sciBert, query_text)).detach().numpy()
    
    #lets do the search and time the process
    st = time.time()
    distances = []
    for em in embeddings :
        
        em = np.expand_dims(em, axis = 0)
        distances.append(compute_cosine_measure(em, embedding_q)[0][0])
        
    top_k_arguments = np.argsort(np.array(distances))[::-1][:top_k]
    et = time.time()
    
    return et - st, top_k_arguments

    
time_cosine, indexes_top = do_consine_search(embeddings, "Middle East Virus", sciBert, 20)
print('Cosine search time :  ', time_cosine, ' seconds')

index_to_title(indexes_top)
        

We got good results from the cosine search over embeddings, Sci-BERT model was able to provide us some satisfactory level of accuracy eventhough it was not fine-tuned with COVID-19 text corpus.
# ### Improving search speed and efficiency with FAISS 
# Cosine Metric search was good enough but it took 11 seconds to provide search results. We can speed it up in many ways, one way is to do batch cosine-distance calculation but still it takes some time. I found this library from facebook, which could do semantic search efficiently with great speed. It builds an index in RAM and uses that index to perform lookups. 
# let's see how to build an index 

# In[ ]:


import faiss


# Build the index

# In[ ]:


n_dimensions = embeddings.shape[1] #Number of dimensions (764)

fastIndex = faiss.IndexFlatL2(n_dimensions) # We will create an index of type FlatL2, there are many kinds of indexes, you can look at it in their repo.
fastIndex.add(embeddings.astype('float32')) # Add the embedding vector to faiss index, it should of dtype 'float32'


# Now we have built  the index, we can perform lookup efficiently.

# In[ ]:


def do_faiss_lookup(fastIndex, query_text, model, top_k):
    n_embeddings = embeddings.shape[0]
    embedding_q = compute_mean(embedding_fn(sciBert, query_text)).detach().numpy()
    
    #let it be float32
    embedding_q = embedding_q.astype('float32')
    
    #perform the search
    st = time.time()
    matched_em, matched_indexes = fastIndex.search(embedding_q, top_k) # it returns matched vectors and thier respective indexes, we are interested only in indexes.
    
    #indexes are already sorted wrt to closest match
    et = time.time()
    
    return et - st, matched_indexes[0]

time_faiss_cpu, indexes_top_faiss = do_faiss_lookup(fastIndex, "Middle East Virus", sciBert, 20)
print('Faiss index lookup time :  ', time_faiss_cpu, ' seconds')

index_to_title(indexes_top_faiss)


    


# Hurray! We have completed Faiss index lookup, it can be noted that results are same as cosine-search , however it took us only **40 milliseconds**, almost 370x speedup. Faiss also supports GPU computations, so we can even build an index on GPU and use GPU cores for computation.
# 
# ### FAISS on GPU
# 
# Let's try to build same thing on GPU. We need to install faiss-gpu

# In[ ]:


import faiss

n_dimensions = embeddings.shape[1] #Number of dimensions (764)

fastIndex_gpu = faiss.IndexFlatL2(n_dimensions) # We will create an index of type FlatL2, there are many kinds of indexes, you can look at it in their repo.

#copy the index to GPU 
res = faiss.StandardGpuResources()

fastIndex_gpu = faiss.index_cpu_to_gpu(res, 0, fastIndex_gpu)

fastIndex_gpu.add(embeddings.astype('float32')) # Add the embedding vector to faiss index, it should of dtype 'float32'


# Creating a GPU index will take soem time. Wait for it. Now you can do search similar to CPU.

# In[ ]:


def do_faiss_lookup_gpu(fastIndex, query_text, model, top_k):
    n_embeddings = embeddings.shape[0]
    embedding_q = compute_mean(embedding_fn(sciBert, query_text)).detach().numpy()
    
    #let it be float32
    embedding_q = embedding_q.astype('float32')
    
    #perform the search
    st = time.time()
    matched_em, matched_indexes = fastIndex.search(embedding_q, top_k) # it returns matched vectors and thier respective indexes, we are interested only in indexes.
    
    #indexes are already sorted wrt to closest match
    et = time.time()
    
    return et - st, matched_indexes[0]

time_faiss_gpu, indexes_top_faiss = do_faiss_lookup_gpu(fastIndex, "Middle East Virus", sciBert, 20)
print('Faiss index lookup time :  ', time_faiss_gpu, ' seconds')

index_to_title(indexes_top_faiss)


# The GPU lookup tool **33ms** which is almost same as CPU based look-up.
# Now let us conclude the time of all three experiments :
# 

# In[ ]:


print('CPU based cosine-distance metric lookup : (Brute-force method : )', time_cosine)
print('CPU based FAISS index lookup : ', time_faiss_cpu)
print('GPU based FAISS index lookup : ', time_faiss_gpu)


# **Downlaod the embeddings of CORD-19 Titles here : [CORD-19 Title embeddings](https://drive.google.com/file/d/1rCA5Y_7gL6Maitcf0_5mjWfYGml3vHEO/view?usp=sharing)**
# 
# Thank you for your time, In case of any queries or corrections, mail me at narasimhaprasannahn@gmail.com
