#!/usr/bin/env python
# coding: utf-8

# #### Create an Embeddings Index
# 
# This Script produces an embedding index that allows a researcher to perform a semantic similarity search on the cord-19 dataset. It has a few features
# 
# - You can search an index (Faiss) for sentences with semantic similarity to an input query. 
#     - The sentences are constructued using sentence embeddings (SentenceTransformer library)
#     - SentenceTransformer is initialised with covid-bert-base, and then fine-tuned with NLI and STS tasks so it adds semantic components to the underlying covid based language model. 
#     
# - A search filter function allows you to limit the documents compiled into the semantic index, by keyword; 
#     - If the keyword appears in the documents precompiled "list of entities", the document is inclued in the index. 
#     - Entities(e.g. compounds, treatments, protocols) are detected with scipacy,
# 
# Briefly, the following steps are performed.  . 
# 
# - Report 
# 
# - Part A
#     - Step 1: Initialise SentenceTransfomer model with covid_bert_base (https://huggingface.co/deepset/covid_bert_base)
#     - Step 2: Fine-tune SentenceTransformer model with natural language inference (sentence entailment) and semantic similarity (sts) tasks
#  
# 
#  - Part B 
# 
#     - This part is divided into Part B.1 and Part B.2
# 
#     - Part B.1 is run on Kaggle kernel. It generates a dataframe with columns paper_id, abstract and body_text.
# 
#     - Part B.2 was run on Google Colab, which connects to a GCP VM, with 16 cores to allow efficient multiprocessin- g of dataframe batches. . I included the code here to view. This part takes the dataframe from B.1, and uses scispacy to extract scientific entities, and creates a new column "ents" in the dataframe. 
# 
# -  Part C 
#  -   Create Faiss Index for each sentence within corpus 
#   
# 
# ## TO DO
# still missing some abstracts which means filtering should look at first seciton of body_text for ents...
# 
# search function should be refactored, nlp(doc) should be factored out so its only performed once for each doc 
# 
# memory usage is too high. Need to rethink use of garbage collector, or more memory efficient table storage
# 
# 
# 
# 

# ## References 
# 
# #### Kaggle Kernels 
# 
# -  https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv
# - https://www.kaggle.com/maksimeren/covid-19-literature-clustering
# 
# 
# These kernels provided code cells cut+pasted into this notebook. They have been to used generate a clean dataset through parsing json files. Many thanks. 
# 
# 
# #### Spacy
# 
# - https://spacy.io/
# 
# Spacy is used to perform nlp pipeline functions such as tokenization, sentence segmentation and span retrieval
# 
# #### deepset/covid-bert-base
# 
# - https://huggingface.co/deepset/covid_bert_base
# - https://github.com/deepset-ai/FARM/blob/master/examples/lm_finetuning.py 
# 
# DeepsetAI script showing how to fine-tune BERT language model with a language modeling task. From what I can gather, they used script lm_finetuning.py in their FARM tools using the CORD-19 dataset to fine-tune the model. 
# 
# #### SentenceTransformers
# - https://github.com/UKPLab/sentence-transformers
# - https://github.com/UKPLab/sentence-transformers/blob/master/examples/training_nli_bert.py
# 
# After loading the covid-bert model, we continued to fine-tune the sentence-transformer model to perform well of natural language tasks, since we will be performing these sorts of tasks when runnning our searches on the embeddings-index. 
# 
# #### Faiss
# - https://gist.github.com/mdouze/e30e8f57a98ed841c082cc68baa14b4a
# This provides code to serialise and deserialise the index so it can be pickled. 

# ### Setup and Installation  

# In[ ]:


# Display full output in Jupyter notebook cell (not only final statement)
from __future__ import print_function
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


# torch from pytorch (huggingface)
import torch
#from transformers import *
import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path

import collections
from tqdm import tqdm
import pprint

tqdm.pandas(desc="my bar!")


# In[ ]:


# if using GPU, test GPU is working

cuda = torch.device('cuda')     # Default CUDA device
torch.__version__
torch.cuda.get_device_name(0)


# In[ ]:


# set display options
pd.options.display.max_rows
pd.set_option('display.max_colwidth', -1)
from IPython.display import display, HTML

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)


# In[ ]:


get_ipython().run_cell_magic('capture', '', '# clone sentence-transformers code and examples, and install\n# install sentence-transformers and download repo to perform fine-tuning steps. pip install will not give you access to all the examples\n\n!git clone https://github.com/UKPLab/sentence-transformers.git\nos.chdir("/kaggle/working/sentence-transformers")\n!pip install -e .')


# In[ ]:


get_ipython().run_cell_magic('capture', '', '# Install spacy and scispacy, and scispacy language model\n!pip install scispacy\n!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz')


# In[ ]:


import scispacy
import spacy
import en_core_sci_sm
nlp = en_core_sci_sm.load()
nlp.max_length=100000000 # for extra long documents


# In[ ]:


#!pip install sentence-transformers
#import os
#os.chdir("/content/sentence-transformers")
from sentence_transformers import SentenceTransformer


# In[ ]:



get_ipython().run_cell_magic('capture', '', '#!pip install faiss-cpu\nuseGPU=True\n\n!pip install faiss-gpu\nimport faiss')


# In[ ]:


# define some generic functions

# define sentence_embedding function
def embed_sentence_list(model,list_of_sents):
    sentence_embeddings=model.encode(list_of_sents)
    doc_matrix=np.asarray(sentence_embeddings,dtype=np.float32)
    return(doc_matrix)


# we need to serialise faiss index to save it to output
#https://gist.github.com/mdouze/e30e8f57a98ed841c082cc68baa14b4a

def serialize_index(index):
    """ convert an index to a numpy uint8 array  """
    writer = faiss.VectorIOWriter()
    faiss.write_index(index, writer)
    return faiss.vector_to_array(writer.data)


def deserialize_index(data):
    reader = faiss.VectorIOReader()
    faiss.copy_array_to_vector(data, reader.data)
    return faiss.read_index(reader)


# ### End of Setup
# 

# ## Report: Semantic Search for limited documents 
# 
# This section contains the report, and assumes Part A, B and C have been run. We can load pre-built objects from those sections from the input directory.
# 

# ### First,** load our fine-tuned SentenceTransformer model (created in Part A below)
# 

# In[ ]:


model_load_path='/kaggle/input/bertcovidbasicnlists/training_nli_sts_covid-bert-base-2020-04-01_00-26-48'

if useGPU:
    model=torch.load(model_load_path) 

else: # not working
    #model=torch.load(model_load_path,map_location=torch.device('cpu')) 
    device = torch.device('cpu')
    model=SentenceTransformer()
    model.load_state_dict(torch.load(model_load_path, map_location=device) # not working
# type(model) #  model is of type "SentenceTransformer"


# ### Second, load our processed CORD-19 dataset (see Part B)
# Format of dataframe:
# 
# - [paper_id, abstract, body_text, ents]
# 
# paper_id - paper_id of the original paper
# abstract - text of the abstract
# body_text - text of the body 
# ents - a list of entities extracted from the abstract (or body_text, if abstract missing) with scipy entity recognition
# 
# See Part B details. Work largely cut+paste cells from: https://www.kaggle.com/maksimeren/covid-19-literature-clustering

# In[ ]:


# Load pre-built files containing dataframes, where each row represents one paper.
# Columns are [paper_id, abstract_text, body_text,ents]
# The "ents" column represents entities present in the abstract (or if not provided, the body text.) 
# The "ents" column was generate using code in Part B, but was run on Google Colab as described below. 
def read_full_cord19_df():
    df_covid1=pd.read_csv("/kaggle/input/cord19-df-with-entities/1.csv",index_col=0)
    df_covid2=pd.read_csv("/kaggle/input/cord19-df-with-entities/2.csv",index_col=0)
    df_covid3=pd.read_csv("/kaggle/input/cord19-df-with-entities/3.csv",index_col=0)
    df_covid4=pd.read_csv("/kaggle/input/cord19-df-with-entities/4.csv",index_col=0)
    df_covid5=pd.read_csv("/kaggle/input/cord19-df-with-entities/5.csv",index_col=0)

    # Concatenate individual dfs to one df
    df_covid_ents=pd.concat([df_covid1,df_covid2,df_covid3,df_covid4,df_covid5])
    return df_covid_ents


# In[ ]:


# Read in processed Document dataframe 
df_covid_ents=read_full_cord19_df()


# ### Third, read in our pre-trained gpu index, and associated dictionary
# 

# In[ ]:


# Load pre-built faiss index from input
# This contains an embedding for each sentence, for each document
# The sentence embedding model has been trained as per Part A, below. 

def read_full_faiss_index():
    filepath=Path("/kaggle/input/faiss-index-file-full-d") / "index_faiss_file_all"
    file=open(filepath, "rb")
    data=pickle.load(file)
    faiss_index=deserialize_index(data)

    # convert index from cpu to gpu
    gpu_index=None
    if useGPU==True:
        res = faiss.StandardGpuResources()  # use a single GPUres = faiss.StandardGpuResources()  # use a single GPU
        gpu_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
    
    # Load pre-built index dictionary
    # format, for each index in faiss index (key), values are the paper_id, and sentence_id, of the embedded sentence at that index locaiton in the faiss index. 

    filepath = Path("/kaggle/input/faiss-index-to-doc-sent-ids-dict") / "faiss_index_ids_dict_all"
    #filepath="/content/drive/My Drive/kaggle/covid19/input/faiss_index_ids_dict_all"

    infile=open(filepath, "rb")
    ids_dict=pickle.load(infile)
    
    return gpu_index, ids_dict, faiss_index


# In[ ]:


gpu_index, ids_dict, faiss_index=read_full_faiss_index()


# ## Test Faiss Index Search with Step by Step Query 
# 
# ### Step 1: Create and Embed Input Query
# Generate a test query for semantic similarity search.
# 
# 
# The faiss engine does an inner product on the query embeddings against the indexed embeddings, so we must normalise the query embeddings first. (The indexed embeddings were normalised upon creation. 
# 

# In[ ]:


# Let us run a test queries
query_list=[]
query_list.append("nurse to patient transmission in aged care facilities")

# create a numpy array, and normalise
xq=embed_sentence_list(model,query_list) # model is instantiated previously from sentence-transformers
faiss.normalize_L2(xq)


# ### Step 2 - Generate Top K Semantic Search Results
# We can now generate the top K semantic matches. This returns the top k best sentences, with the closest semantic meaning to our query sentence. 
# 
# The object "top_k" is returned as a list with 2 elements. The first element contains the cosine simmilarity scores of the matches, and the second element contains the ids (row number) of the match in the faiss index. 
# 
# It returns one row per query. Within each row, columns are the 1..k'th best match for the query.

# In[ ]:


top_k = gpu_index.search(xq[0,:].reshape(1,-1), 3) # sanity check
top_k


# * ### Step 3 Examine the Results
# 
# We can examine the scores and index of the top sentence matches, returned from the search of faiss. 
# However, we still need to retrieve details of the original sentence, to see whether our match makes any sense. 

# In[ ]:


j=0 # first result

#  return faiss index values from matches from second element in result list (at top_k[1]).
# Once in list, get first row (query), and j'th column (j'th best match ) (at ...to_list()[0][j])
index_tmp=top_k[1].tolist()[0][j] 

# Retrieve original sentence text from best match 
paper_id, sent_id=ids_dict[index_tmp][0] # use dictionary to retrieve original paper, and sentence location 
paper_row=df_covid_ents[df_covid_ents['paper_id']==paper_id] # get row in document by filtering for unique paper id 

doc=nlp(paper_row.iloc[0]['body_text']) # convert text to sentences 


# get sentence - need to re-execute spacy pipeline to retrieve sentences since this is not stored. 
list_of_sents=[sent.text for sent in doc.sents]
sent=list_of_sents[sent_id] # retrieve sentence id


print(sent)


# That's great. The sentence looks relevant. 
# 
# But we might need more context to know whether that sentence is relevant to our query. We use spacy spans to grab the context, i.e. text around the sentence. 

# #### Step 4, Generate Sentence "Context"

# In[ ]:


list_of_spans=[sent for sent in doc.sents]

span_start=list_of_spans[sent_id].start
span_end=list_of_spans[sent_id].end
if span_start < 100:
    span_start = 0
else:
    span_start -=100
if (span_end + 100) > len(doc):
    span_end = len(doc)
else:
    span_end += 100
    
span_results=doc[span_start: span_end]


list_of_spans[sent_id] # original sentence 
span_results # full span 


# ### Step 5, Double-check cosine similiarity score manually 
# 
# Let's check the score returned by faiss manually. It should be the cosine between the embedded query matrix, and the result vector. 
# 

# In[ ]:


score_tmp=top_k[0].tolist()[0][j] # return faiss scores from first element in result list (at top_k[0])
score_tmp # faiss inner product score

# check the score manually, to make sure it is what we expect 
doc_matrix=embed_sentence_list(model,list_of_sents)
faiss.normalize_L2(doc_matrix)
from scipy.spatial.distance import cosine
1-cosine(xq[0,:],doc_matrix[sent_id,:]) # first query (this is a cosine distance, not a cosine similarity)  # agreement! 

np.inner(xq[0,:],doc_matrix[sent_id,:]) # agreement! 


# ### Functions to generate new index based on a subset of documents
# #### Filtered by Keyword
# 
# Now, we might need to create our own index, based on a limited set of keywords. 
# This is a bit time-consuming, and ideally I would shift this to a GCP VM, with an API where it might run more quickly.

# In[ ]:


pp = pprint.PrettyPrinter(indent=4)

def create_gpu_index(df_covid_tmp):
    """ 
    used to generate a smaller faiss index, limited by keyword
    """
    tqdm.pandas(desc="my bar!")

    print("creating new gpu index of size", df_covid_tmp.shape[0])
    
    # init faiss index
    d=768 # sentence transformer embedding length
    res = faiss.StandardGpuResources()  # use a single GPU
    index = faiss.IndexIDMap(faiss.IndexFlatIP(d)) # IP is inner product. Data must be normalised first
    gpu_index_tmp = faiss.index_cpu_to_gpu(res, 0, index)

    # init dict
    ids_dict_tmp = collections.defaultdict(list)

    ids_next=0
    i=0
    for row in tqdm(range(df_covid_tmp.shape[0])):
        doc=nlp(df_covid_tmp.iloc[row]['body_text'])
        paper_id=df_covid_tmp.iloc[row]['paper_id']
        
        list_of_sents=[sent.text for sent in doc.sents]
        #if len(list_of_sents) > 800:
        #    list_of_sents=list_of_sents[:800] # just taking first 800 sentences for memory reasons. 
            
        doc_matrix=embed_sentence_list(model,list_of_sents)
        faiss.normalize_L2(doc_matrix)

        # ids in faiss index begin after last idx (last document processed, last sentence)
        custom_ids = np.array(range(ids_next, ids_next+len(doc_matrix))) # from last postion (range add +1) to new position
        ids_next=ids_next+len(doc_matrix) # increment by current document length. Current doc lenght = num sentences in current doc             
        gpu_index_tmp.add_with_ids(doc_matrix, custom_ids)
        for sent_idx, faiss_ids_val in enumerate(custom_ids): # sentence_idx is sentence id within the 1 document, faiss_ids_val is the faiss index value
            items=(paper_id, sent_idx)
            ids_dict_tmp[faiss_ids_val].append(items)
            
    del doc_matrix
    del doc
    gc.collect()
    return(gpu_index_tmp,ids_dict_tmp)


# In[ ]:


# to  do - add in function to process search response
# combine with absolute match result to ensure sentence contain keywords
pp = pprint.PrettyPrinter(indent=4)
import gc
def search(model,query, k=3, gpu_index_default=None,ids_dict_default=None,keyword=None,df_covid_ents_default=None):
    """
    returns results based on keyword, the gpu_index, and the ids_dict
    
    :param model, query, k, keyword, gpu_index_default, ids_dict_default 
    :type
    :param query
    :type 
    """
         
    query_list=[query]
    xq=embed_sentence_list(model,query_list) # model is instantiated previously from sentence-transformers
    faiss.normalize_L2(xq)

    if df_covid_ents_default is None:
      df_covid_ents=read_full_cord19_df() #df_covid_ents
    else:
      df_covid_ents=df_covid_ents_default

    if gpu_index_default==None or ids_dict_default==None:

      if keyword is not None:
        df_covid_tmp=df_covid_ents[df_covid_ents['ents'].str.contains(keyword, na=False) ].copy() # filter on keyword
        print("Create new faiss index based on", df_covid_tmp.shape[0], "documents")
        gpu_index_tmp,ids_dict_tmp = create_gpu_index(df_covid_tmp)  # generate new index, based on filtered dataframe

      else: # no keyword but still need to create index from scratch
        gpu_index_tmp, ids_dict_tmp = read_full_faiss_index() #gpu_index_default # if no keyword, use original artifacts
        df_covid_tmp=df_covid_ents #df_covid_ents

    else: 
        print("using custom index and dict - ignoring any keywords") #ignore keyword
        gpu_index_tmp=gpu_index_default
        ids_dict_tmp=ids_dict_default
        df_covid_tmp=df_covid_ents
  
    # get top k best matches through index look_up
    top_k = gpu_index_tmp.search(xq, k)  
    # >>>
    pp.pprint("end faiss search")
    # >>>
      
    # init output results dataframe 
    colnames=["query","sentence","score", "span","paper_id"]    
    results=pd.DataFrame(columns=colnames)
      
    for i, _id in tqdm(enumerate(top_k[1].tolist()[0])): # for each result in the top k, element zero since only one query (could do batch queries)
        paper_id, sent_id=ids_dict_tmp[_id][0]
          
        ## retrieve sent value   
        # get row in document by filtering for unique paper id 
        paper_row=df_covid_tmp[df_covid_tmp['paper_id']==paper_id]
        
        # convert text to sentences 
        doc=nlp(paper_row.iloc[0]['body_text'])

        # get sentence - need to re-execute spacy pipeline to retrieve sentences since this is not stored. 
        list_of_sents=[sent.text for sent in doc.sents]
        sentence_result=list_of_sents[sent_id] # retrieve original sentence text       

        # retrieve spans
        list_of_spans=[sent for sent in doc.sents]
        
        span_start=list_of_spans[sent_id].start
        span_end=list_of_spans[sent_id].end

        if span_start < 100:
          span_start = 0
        else:
          span_start -= 100
                
        if (span_end + 100) > len(doc):
          span_end = len(doc)
        else:
          span_end += 100
                
        span_results=doc[span_start: span_end]        

        # get score
        score_tmp=top_k[0].tolist()[0][i] # score for i'th match
            
        #title=paper_row['title'] # to do
        #authors=paper_row['authors'] # to 
        #abstract=paper_row['abstract'] # to do
            
        tmp=pd.DataFrame( [pd.Series([query,sentence_result, score_tmp, span_results, paper_id],index=colnames)] )
        results=results.append(tmp, ignore_index=True, sort=False)
              
    return(results, gpu_index_tmp, ids_dict_tmp)


# 
# ## Example: Simple Semantic Search 
# 
# Lets now do a query on all the data (full index) for sentences that shed light on nurse transmission to patients in hospitals. 

# In[ ]:


# if GPU is working, run search 

results=search(model=model,
               query="nurses transmit to patients in hospitals", 
               k=6)
df,index_,dict_=results


# In[ ]:


df=pd.read_csv("/kaggle/input/results-output/full_dataset_results_transmission.csv") # pregenerate results

df[["paper_id","sentence","score","span"]].head() 


# ### Example - Keyword Filtered Document Set - Semantic Similarity Search 
# Now, lets do a refined search. Lets limit the search to only those documents that contain "hydroxychloroquine" within the extracted list of "entities". This will give < 100 documents. We can then compile a limited faiss index from this shortlist of documents. **

# In[ ]:


# works if Kaggle GPU is working
results=search(model=model,
               query="timing intervals of treatment delivered",
               k=3, 
               gpu_index_default=None, 
               ids_dict_default=None,
                 keyword="hydroxychloroquine", 
)
df_results_hcq, gcu_index_hcq, ids_dict_hcq=results


# In[ ]:


# read in results compiled on Google Colab manually for now (since Kaggle GPU unavailable)
df_results_hcq=pd.read_csv("/kaggle/input/results-output/df_results_hcq.csv")
df_results_hcq[['paper_id','sentence','score','span']]


# In[ ]:


# NOTE - we could also run, passing in a pre-compiled index for a given keyword ("hydroxychloroquine") in this case. 
results=search(model=model,
               query="the treatment was delivered at timing intervals ",
               k=3, 
               gpu_index_default=gcu_index_hcq, 
               ids_dict_default=ids_dict_hcq,
               keyword=None, df_covid_ents_default=df_covid_ents
)


# In[ ]:


df_results_hcq_timing, _, _ = results
df_results_hcq_timing[['paper_id','sentence','score','span']]


# ### Part A - SentenceTransformer model fine-tuning 

# In[ ]:


# import sentence_transformer functions 
# See fine-tuning example at - https://github.com/UKPLab/sentence-transformers/blob/master/examples/training_nli_bert.py
os.chdir("/kaggle/working/sentence-transformers")

from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.readers import *
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.datasets import *

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data import DataLoader
import math
import logging
from datetime import datetime


# Deepset have finetuned a BERT language model on the CORD-19 dataset (see references above). 
# 
# https://huggingface.co/deepset/covid_bert_base
# 
# We load this model. 
# The following code is otherwise lifted from "https://github.com/UKPLab/sentence-transformers/blob/master/examples/training_nli_bert.py" to complete the remainder of the fine-tuning. 
#     

# In[ ]:


# set up SentenceTransformer model, using pooled averaging. 
# start with covid_bert_base model, rather than CORD-19 task

word_embedding_model = models.BERT("deepset/covid_bert_base")

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# In[ ]:


# Now fine-tune model further on NLI and STS tasks to get natural language and semantic tones included in embeddings

get_ipython().system('python examples/datasets/get_data.py # downloads AllNLI.zip and STSbenchmark.zip datasets ')


# In[ ]:


os.chdir("./examples") # switch to examples directory where output is stored
nli_reader = NLIDataReader('datasets/AllNLI')
train_data = SentencesDataset(nli_reader.get_examples('train.gz'), model=model)
model_name = 'covid-bert-base'
batch_size = 16
nli_reader = NLIDataReader('datasets/AllNLI')
sts_reader = STSDataReader('datasets/stsbenchmark')
train_num_labels = nli_reader.get_num_labels()
model_save_path = 'kaggle/working/training_nli_'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)

logging.info("Read STSbenchmark dev dataset")
dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

# Configure the training
num_epochs = 1

warmup_steps = math.ceil(len(train_dataloader) * num_epochs / batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )


# In[ ]:


# save model 
model_save_name='training_nli_covid_'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename=Path("/kaggle/working/" ) / model_save_name 
outfile = open(filename,'wb')
torch.save(model, outfile)


# ### Part B - Create df_covid 
# 
# This part is divided into Part B.1 and Part B.2
# 
# Part B.1 was run on Kaggle kernel. It generates a dataframe with columns paper_id, abstract and body_text.
# 
# Part B.2 was run on Google Colab, which connects to a GCP VM, with 16 cores to allow efficient multiprocessing of dataframe batches. . I included the code here to view. This part takes the dataframe from B.1, and uses scispacy to extract scientific entities, and creates a new column "ents" in the dataframe. 

# In[ ]:


#https://www.kaggle.com/ivanegapratama/covid-eda-initial-exploration-tool
# papers_df only got stuff from biox...
# we need to get the full data set 

#https://www.kaggle.com/ivanegapratama/covid-eda-initial-exploration-tool
    
#import matplotlib.pyplot as plt
#plt.style.use('ggplot')
import glob
import json

root_path = '/kaggle/input/CORD-19-research-challenge'
metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})
#meta_df.head()


# In[ ]:


all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
len(all_json)


# In[ ]:


class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            if 'abstract' in content.keys():
                for entry in content['abstract']:
                    self.abstract.append(entry['text'])
            else:
                self.abstract.append('')
            # Body text
            if 'body_text' in content.keys():
                for entry in content['body_text']:
                    self.body_text.append(entry['text'])
            else:
                self.body_text.append('')
                
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)

            # Extend Here
            #
            #
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'
first_row = FileReader(all_json[0])
print(first_row)



# In[ ]:


dict_ = {'paper_id': [], 'abstract': [], 'body_text': []}
for idx, entry in enumerate(all_json):
    if idx % (len(all_json) // 10) == 0:
        print(f'Processing index: {idx} of {len(all_json)}')
    content = FileReader(entry)
    dict_['paper_id'].append(content.paper_id)
    dict_['abstract'].append(content.abstract)
    dict_['body_text'].append(content.body_text)
df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text'])
#df_covid.head()


# In[ ]:


# https://www.kaggle.com/maksimeren/covid-19-literature-clustering

dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}
for idx, entry in enumerate(all_json):
    if idx % (len(all_json) // 10) == 0:
        print(f'Processing index: {idx} of {len(all_json)}')
    content = FileReader(entry)
    
    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    # no metadata, skip this paper
    if len(meta_data) == 0:
        continue
    
    dict_['paper_id'].append(content.paper_id)
    dict_['abstract'].append(content.abstract)
    dict_['body_text'].append(content.body_text)
    
    # also create a column for the summary of abstract to be used in a plot
    if len(content.abstract) == 0: 
        # no abstract provided
        dict_['abstract_summary'].append("Not provided.")
    elif len(content.abstract.split(' ')) > 100:
        # abstract provided is too long for plot, take first 300 words append with ...
        info = content.abstract.split(' ')[:100]
        summary = get_breaks(' '.join(info), 40)
        dict_['abstract_summary'].append(summary + "...")
    else:
        # abstract is short enough
        summary = get_breaks(content.abstract, 40)
        dict_['abstract_summary'].append(summary)
        
    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    
    try:
        # if more than one author
        authors = meta_data['authors'].values[0].split(';')
        if len(authors) > 2:
            # more than 2 authors, may be problem when plotting, so take first 2 append with ...
            dict_['authors'].append(". ".join(authors[:2]) + "...")
        else:
            # authors will fit in plot
            dict_['authors'].append(". ".join(authors))
    except Exception as e:
        # if only one author - or Null valie
        dict_['authors'].append(meta_data['authors'].values[0])
    
    # add the title information, add breaks when needed
    try:
        title = get_breaks(meta_data['title'].values[0], 40)
        dict_['title'].append(title)
    # if title was not provided
    except Exception as e:
        dict_['title'].append(meta_data['title'].values[0])
    
    # add the journal information
    dict_['journal'].append(meta_data['journal'].values[0])
    
df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal', 'abstract_summary'])
df_covid.head()


# In[ ]:


df_covid1=df_covid.drop( ["ents"],axis=1)


# In[ ]:


# export index
from pathlib import Path
filepath = Path("/kaggle/working") /"df_covid"
outfile=open(filepath, "wb")
pickle.dump(df_covid,outfile)


# In[ ]:


# Download model. Upload to datasets
os.chdir("/kaggle/working")
from IPython.display import FileLink
FileLink(r'df_covid')


# ### Part C
# Part C involves embedding the CORD-19 dataset using our fine-tuned SentenceTranformer model (from Part A), and setting up the faiss  index to store the embeddings.

# In[ ]:


os.chdir("/kaggle/working/sentence-transformers")

from sentence_transformers import SentenceTransformer
os.chdir("/kaggle/working/")


# In[ ]:


# Load model from input 
model_load_path='/kaggle/input/bertcovidbasicnlists/training_nli_sts_covid-bert-base-2020-04-01_00-26-48'
model=torch.load(model_load_path) 
# type(model) #  model is of type "SentenceTransformer"


# In[ ]:


# init faiss index
import faiss # in case not already loaded 
d=768 # sentence transformer embedding length
res = faiss.StandardGpuResources()  # use a single GPU
index = faiss.IndexIDMap(faiss.IndexFlatIP(d)) # IP is inner product. Data must be normalised first
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)


# In[ ]:


# load covid_df
# import dictionary
from pathlib import Path
filepath = Path("/kaggle/input/covid-docs-processed-dataframe") / "df_covid_ents"
infile=open(filepath, "rb")
df_covid=pickle.load(infile)


# In[ ]:


df_covid.shape


# In[ ]:


# embed documents and add them to faiss index 
# maintain a dictionary which stores, for each index in faiss, details of the paper (from papers_df)

# index - ids, paper_id, sent_idx

ids_dict = collections.defaultdict(list)

ids_next=0

#for row in range(papers_df.shape[0]):
for row in range(df_covid.shape[0]):

    # doc=nlp(papers_df.iloc[row]['text'])
    doc=nlp(df_covid.iloc[row]['body_text'])

    # paper_id=papers_df.iloc[row]['paper_id']
    paper_id=df_covid.iloc[row]['paper_id']

    list_of_sents=[sent.text for sent in doc.sents]
    doc_matrix=embed_sentence_list(model,list_of_sents)
    faiss.normalize_L2(doc_matrix)

    # ids in faiss index begin after last idx (last document processed, last sentence)
    custom_ids = np.array(range(ids_next, ids_next+len(doc_matrix))) # from last postion (range add +1) to new position
    ids_next=ids_next+len(doc_matrix) # increment by current document length. Current doc lenght = num sentences in current doc             
    gpu_index.add_with_ids(doc_matrix, custom_ids)
    for sent_idx, faiss_ids_val in enumerate(custom_ids): # sentence_idx is sentence id within the 1 document, faiss_ids_val is the faiss index value
        items=(paper_id, sent_idx)
        ids_dict[faiss_ids_val].append(items)


# In[ ]:





# In[ ]:


# serialize index for output 
cpu_index=faiss.index_gpu_to_cpu(gpu_index)
index_ser=serialize_index(cpu_index)


# In[ ]:


# export index
from pathlib import Path
filepath = Path("/kaggle/working") /"index_faiss_file"
outfile=open(filepath, "wb")
pickle.dump(index_ser,outfile)


# In[ ]:


# export dictionary
from pathlib import Path
filepath = Path("/kaggle/working") /"faiss_index_ids_dict"
outfile=open(filepath, "wb")
pickle.dump(ids_dict,outfile)


# In[ ]:


# Download model. Upload to datasets
os.chdir("/kaggle/working")
os.listdir()
from IPython.display import FileLink
FileLink(r'index_faiss_file')
FileLink(r'faiss_index_ids_dict')

