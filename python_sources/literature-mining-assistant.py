#!/usr/bin/env python
# coding: utf-8

# # Exploring the CORD dataset: mining insight from a mountain of literature
# 
# The speed at which COVID-19 has spread across the globe has presented a unique challenge for healthcare researchers and practitioners. The scientific community must act quickly to understand this virus and the pathology it causes so that we can develop treatments, preventative public health measures and, ultimately, a vaccine. Rapidly moving science, however, presents a new problem: a massive amount of literature produced in a short period of time such that no individual researcher can effectively digest it all with the speed required to quickly take the next step.
# 
# To help with this problem, I will attempt to build a literature mining assistant. My process for literature reading typically is as follows:
# 
# 1. Run a search for papers relevant to my topic of interest
# 2. Read abstracts of search results and select most relevant papers for detailed reading
# 3. Thoroughly read those papers to understand their conlcusions
# 4. While reading, make note of references with seemingly useful further information
# 5. Back to step 2 with references from step 4
# 
# To accelerate this process, I will use a combination of search tools and abstractive summarization.
# 
# To facilitate step 1, I will use the open-source search engine library [Pyserini](https://github.com/castorini/pyserini). Pyserini is a Python wrapper for [Anserini](http://github.com/castorini/Anserini), which uses the [Apache Lucene](https://lucene.apache.org/) open source search engine framework to build and search indexes from large text databases. The Anserini team has already build an index of the CORD dataset, which I will use here (thanks, Anserini team!).
# 
# To accelerate steps 2 and 3, I will use an abstractive summarization approach. The researcher will choose a paper for summarization and the algorithm will then produce a summary of the most relevant information in the paper.
# 
# The release of the transformer model [BERT](http://arxiv.org/abs/1810.04805) has been lauded as the "ImageNet moment" of the NLP world, showing advances by almost every available metric over the previous state of the art. The model, or now set of models, is a large transformer pre-trained using a masking technique rather than typical left to right sequence learning, allowing it to learn both left to right and right to left sequence dependencies. Here I will apply the smaller, notebook-friendly distilBERT model to reduce each paragraph to a lower-dimensional vector and then find the most relevant paragraphs using [cosine similarity](https://towardsdatascience.com/nlp-text-similarity-how-it-works-and-the-math-behind-it-a0fb90a05095). 
# 
# One of the most exciting model architectures in modern machine learning are denoising autoencoders. The success of the autoencoder approach in predicting animal behavior given a set of neural recordings was what first inspired me to explore the world of machine learning and data science ([Pandarinath et al](https://www.biorxiv.org/content/10.1101/152884v1)). The denoising autoencoder BART is a wonderful model for abstractive summarization, I will use it here to generate summaries of the relevant paragraphs from each search result.
# 
# ### Justification
# Here I have chosen to build a literature mining assistant. You might call it a human-majorly-in-the-loop algorithm. I have chosen to keep the researcher at the heart of the process as opposed to building a semantic understanding or question answering model. The power of the human mind is to make connections between loosely associated ideas, many of which will not be contained in the CORD dataset. No NLP model is yet capable of building the world-level model required to make these connections, although Yoshua Bengio's [world scope](https://medium.com/syncedreview/new-study-tracks-nlp-development-and-direction-through-a-world-scope-bf8023a6588f) project is aiming to do just that. If these efforts are successful the application of these models to scientific literature searches could significantly increase the efficiency of scientific research, or possibly synthesize a host of latent ideas already existing in the literature. For now, Biologists are still the best tool we have for real semantic understanding of biological literature.
# 
# ### Results
# The tool works pretty well! The summaries are informative and sometimes have bits of useful unexpected information, like references to other papers or data. BERT and BART work wonderfully and the Huggingface Transformers package provides a great interface for them for Pytorch. The processing time is a bit long, but could easily be accelerated outside of the restrictions of the kaggle notebook environment. This tool could be used by researchers to increase the efficiency of their reading by quickly pulling out the real meat of the most relevant papers quickly.
# 
# ### Future directions:
# 
# If this were my job and I could spend more time on this project, I would love to accelerate steps 4 and 5. I would do this by detecting the references present in the relevant paragraphs, pulling full text from Pub Med, and repeating the BART/BERT summarization process. It would be nice to rework the search to allow a PubMed search initially to expand the usefulness of this tool beyond COVID-19 research.
# 
# Credit:
# 
# My work here is primarily for my own edification and I owe many thanks to the teams which have provided most of the libraries and code that have allowed me to quickly complete such a fun project. Thanks to the Anserini team for generating the Lucene index for the CORD dataset. Thanks to the authors of the BERT and BART papers and the [Huggingface transformers](https://huggingface.co/transformers/) team for creating a wonderful python interface for these models. Much credit to Kaggle user Dirk the Engineer for their much more thorough [notebook](https://www.kaggle.com/dirktheeng/anserini-bert-squad-for-semantic-corpus-search) using Anserini and BERT in a question-answering approach, which provided the primary inspiration for this work. For a much more serious attempt at completing the goals of this competition, be sure to check out their work. 

# # Import libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Import HTML library for displaying results nicely
from IPython.core.display import display, HTML
# json reader
import json
# tqdm progress meter
from tqdm import tqdm

# huggingface transformers
get_ipython().system('pip install transformers')
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import BartTokenizer, BartForConditionalGeneration


# # Set up openJDK for running Pyserini

# In[ ]:


get_ipython().system('curl -O https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz')
get_ipython().system('mv openjdk-11.0.2_linux-x64_bin.tar.gz /usr/lib/jvm/; cd /usr/lib/jvm/; tar -zxvf openjdk-11.0.2_linux-x64_bin.tar.gz')
get_ipython().system('update-alternatives --install /usr/bin/java java /usr/lib/jvm/jdk-11.0.2/bin/java 1')
get_ipython().system('update-alternatives --set java /usr/lib/jvm/jdk-11.0.2/bin/java')
os.environ["JAVA_HOME"] = "/usr/lib/jvm/jdk-11.0.2"


# # Install Pyserini and download the Lucene index

# In[ ]:


get_ipython().system('pip install pyserini==0.8.1.0')
from pyserini.search import pysearch
# import lucene index (thanks, anserini team!)
get_ipython().system('wget https://www.dropbox.com/s/d6v9fensyi7q3gb/lucene-index-covid-2020-04-03.tar.gz')
get_ipython().system('tar xvfz lucene-index-covid-2020-04-03.tar.gz')


# # Define functions for running queries and displaying results

# In[ ]:


COVID_INDEX = 'lucene-index-covid-2020-04-03/'

def show_query(query):
    """HTML print format for the searched query"""
    return HTML('<br/><div style="font-family: Times New Roman; font-size: 20px;'
                'padding-bottom:12px"><b>Query</b>: '+query+'</div>')

def show_document(idx, doc):
    """HTML print format for document fields"""
    have_body_text = 'body_text' in json.loads(doc.raw)
    body_text = ' Full text available.' if have_body_text else ''
    return HTML('<div style="font-family: Times New Roman; font-size: 18px; padding-bottom:10px">' + 
               f'<b>Document {idx}:</b> {doc.docid} ({doc.score:1.2f}) -- ' +
               f'{doc.lucene_document.get("authors")} et al. ' +
             # f'{doc.lucene_document.get("journal")}. ' +
             # f'{doc.lucene_document.get("publish_time")}. ' +
               f'{doc.lucene_document.get("title")}. ' +
               f'<a href="https://doi.org/{doc.lucene_document.get("doi")}">{doc.lucene_document.get("doi")}</a>.'
               + f'{body_text}</div>')

def show_query_results(query, searcher, top_k=10):
    """HTML print format for the searched query"""
    hits = searcher.search(query)
    display(show_query(query))
    for i, hit in enumerate(hits[:top_k]):
        display(show_document(i+1, hit))
    return hits[:top_k] 

searcher = pysearch.SimpleSearcher(COVID_INDEX)


# # Run query over Lucene index, display top hits

# In[ ]:


# Enter query here
query = ('COVID-19 incubation period in humans')

hits = show_query_results(query, searcher, top_k=10)


# # Set up pre-trained distilBERT model

# In[ ]:


model_version = 'distilbert-base-uncased'
do_lower_case = True
model = DistilBertModel.from_pretrained(model_version)
tokenizer = DistilBertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)


# # Function for extracting states from text using distilBERT

# In[ ]:


def extract_distilbert(text, tokenizer, model):
    # Convert text to IDs with special tokens specific to the distilBERT model
    text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    text_words = tokenizer.convert_ids_to_tokens(text_ids[0])[1:-1]

    n_chunks = int(np.ceil(float(text_ids.size(1))/510))
    states = []
    
    for ci in range(n_chunks):
        text_ids_ = text_ids[0, 1+ci*510:1+(ci+1)*510]            
        text_ids_ = torch.cat([text_ids[0, 0].unsqueeze(0), text_ids_])
        if text_ids[0, -1] != text_ids[0, -1]:
            text_ids_ = torch.cat([text_ids_, text_ids[0,-1].unsqueeze(0)])
        
        with torch.no_grad():
            state = model(text_ids_.unsqueeze(0))[0]
            state = state[:, 1:-1, :]
        states.append(state)

    state = torch.cat(states, axis=1)
    return text_ids, text_words, state[0]


# # Extract states from the query

# In[ ]:


query_ids, query_words, query_state = extract_distilbert(query, tokenizer, model)


# # Extract states from each paragraph in a specific hit

# In[ ]:


my_fav_hit = 6
doc_json = json.loads(hits[my_fav_hit].raw)

paragraph_states = []
for par in tqdm(doc_json['body_text']):
    state = extract_distilbert(par['text'], tokenizer, model)
    paragraph_states.append(state)


# # Build a cosine similarity matrix between the query and each paragraph in the hit

# In[ ]:


# Compute similarity given the extracted states from sciBERT
def cross_match(state1, state2):
    state1 = state1 / torch.sqrt((state1 ** 2).sum(1, keepdims=True))
    state2 = state2 / torch.sqrt((state2 ** 2).sum(1, keepdims=True))
    sim = (state1.unsqueeze(1) * state2.unsqueeze(0)).sum(-1)
    return sim


# In[ ]:


# Compute similarity for each paragraph
sim_matrices = []
for pid, par in tqdm(enumerate(doc_json['body_text'])):
    sim_score = cross_match(query_state, paragraph_states[pid][-1])
    sim_matrices.append(sim_score)


# # Find indices of most relevant paragraphs

# In[ ]:


paragraph_relevance = [torch.max(sim).item() for sim in sim_matrices]

# Select the index of top 5 paragraphs with highest relevance
rel_index = np.argsort(paragraph_relevance)[-5:][::-1]


# In[ ]:


def show_sections(section, text):
    """HTML print format for document subsections"""
    return HTML('<div style="font-family: Times New Roman; font-size: 18px; padding-bottom:10px; margin-left: 15px">' + 
        f'<b>{section}</b> -- {text.replace(" ##","")} </div>')

display(show_query(query))
display(show_document(my_fav_hit, hits[my_fav_hit]))
for ri in np.sort(rel_index):
    display(show_sections(doc_json["body_text"][ri]['section'], " ".join(paragraph_states[ri][1])))


# # Summarize most relevant paragraphs with BART abstractive summarization

# In[ ]:


text = []
for ri in np.sort(rel_index):
    text = text + paragraph_states[ri][1]


# In[ ]:


torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

sum_tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')
sum_model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
sum_model.to(torch_device)
# set to evaluation mode for speed and memory saving
sum_model.eval()


# In[ ]:



article_input_ids = sum_tokenizer.batch_encode_plus([tokenizer.convert_tokens_to_string(text)], return_tensors='pt', max_length=1024)['input_ids'].to(torch_device)

summary_ids = sum_model.generate(article_input_ids,
                             num_beams=4,
                             length_penalty=2.0,
                             max_length=1000,
                             no_repeat_ngram_size=3)

summary_txt = sum_tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

HTML('<div style="font-family: Times New Roman; font-size: 18px; padding-bottom:10px; margin-left: 15px">' + 
        f'<p><b>Query:</b> {query}</p> &nbsp; <p><b>Summary of results:</b> {summary_txt}</div></p>')

