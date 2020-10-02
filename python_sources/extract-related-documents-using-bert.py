#!/usr/bin/env python
# coding: utf-8

# **CORD-19 Extract Related Documents using Bert**
# * This script introduces useful functions to extract most related papers or paragraphs based on a query using the STOA NLP pretrained models, Bert.
# * This task is regarded as Next-Sequence predication. The query is the first sequence and the paper or paragraph is the second sequence. Bert can generate a sequence-relation score to evaluate the possibility that the second sequence is the next sequence of the first sequence.
# * Given one query, we test each paper or paragraph and rank the sequence-relation scores. The higher the score, the more related the paper with the query.

# **Load Package**
# * For Bert, we use the huggingface implementation from "https://github.com/huggingface/transformers".
# * I use Pytorch as the deep learning framework and tensor process.
# * tqdm is a good process bar.

# In[ ]:


import json
import os
from transformers import *
import torch
from tqdm import tqdm


# **Process Data**
# * In the dataset, there are four folders and each folder contains a list of json files. Each json file has the information of a document.
# * To simplify the processing later, we organize the list of json files of documents in a single folder to one json file. 
# * The output json file is a list, each item is a dictionary for a document.

# In[ ]:


def create_docs_json (data_folder, save_path=None, save_file_name='full_docs.json'):
    #################################################################
    # Create json file for whole docs and save it to current path
    #
    # input: 
    # data_folder - the data folder path contains paper json files
    # 
    # output: 
    # full_docs - a list of dictionary, each dct has the paper info
    #################################################################

    full_docs = []
    for jsonfile in os.listdir(data_folder):
        paper_entity = {}
        with open(os.path.join(data_folder, jsonfile), 'r', encoding='utf8') as f:
            paper = json.load(f)
        paper_entity['paper_id'] = paper['paper_id']
        paper_entity['title'] = paper['metadata']['title']
        abstract = paper['abstract']
        abstract = [item['text'].strip() for item in abstract]
        abstract = ' '.join(abstract)
        paper_entity['abstract'] = abstract
        body_text = paper['body_text']
        paper_entity['body_text'] = body_text
        full_docs.append(paper_entity)
    if save_path is not None:
        save_file = os.path.join(save_path, save_file_name)
    else:
        save_file = save_file_name
    with open(save_file, 'w', encoding='utf8') as wf:
        json.dump(full_docs, wf)
    return full_docs


# **Example**
# * Processing all documents in biorxiv_medrxiv foldere into one json file

# In[ ]:


# Example:
full_docs_biorxiv = create_docs_json("/kaggle/input/CORD-19-research-challenge/2020-03-13/biorxiv_medrxiv/biorxiv_medrxiv", save_file_name='full_docs_full_docs_biorxiv.json')
print("There are " + str(len(full_docs_biorxiv)) + " documents organized into one json file.")


# **Extract Related Documents Using Bert**
# * In this script, I did not use mini-batch, i.e. Each document is a input data, to process the data because Bert is a large model. People may need to use different batch size for different hardware setting. You can modify the functions with mini-batch to speed up the extraction.

# **Functions**
# * text_similarity_score: Compare the query text and candidate candidate and compute the sequence-relation score. The model is Bert model, you can either use pre-trained Bert model or fine-tuning by yourself. The input for the Bert model is the concatenation of query and candidate. We add a special token "[SEP]" between query and candidate to facilite Bert understand the sequence better. As one of the most basic Bert model "bert-base-uncased" has a max-sequence length 512, we will sequence input to length 512. Therefore, you might need to find interesting queries for various questions and interesting text to using this. 
# * extract_similar_papers: Given a single query and the list of documents returned from function "create_docs_json" (you can either use the return value or load from saved json), this function computes the score between query with each document in the list and rank them from high to low. This function returns the top N ranked documents (you can set N with the parameter num_rets). In this function, I use the title+abstract as the candidate for each paper/document. Query will be compared with the title+abstract. You can find your own way to represent the informaiton of a paper/document.
# * extract_similar_txtsegs: Similar as the previous function, but extract text paragraphs or segments instead of documents. Given a single query and the list of documents returned from function "create_docs_json" (you can either use the return value or load from saved json), this function computes the score between query with each paragraphs in all documents and rank them from high to low. This function returns the top N ranked paragraphs (you can set N with the parameter num_rets).

# In[ ]:


def text_similarity_score (query, candidate, model, tokenizer, max_seq_length=512):
    #################################################################
    # Compute similarity score between query text and candidate text
    # input: 
    # query - the query text for comparison
    # candidate - candidate text for comparison
    # model - comparison model
    # tokenizer - tokenizer for tokenzing the plain text
    #
    # output: 
    # txt_sim_score - similarity score
    #################################################################

    txt = query.strip() + "[SEP]" + candidate.strip()
    input_ids = torch.tensor(tokenizer.encode(txt, add_special_tokens=True, max_length=max_seq_length)).unsqueeze(0)
    outputs = model(input_ids)
    txt_sim_score = outputs[0][0][0].item()
    return txt_sim_score


def extract_similar_papers (query, data_dct, model, tokenizer, num_rets=30):
    #################################################################
    # Compute Extract most related papers for the query
    # input: 
    # query - the query text for extraction
    # data_dct - data structure contians all paper information
    # model - similarity comparison model
    # tokenizer - tokenizer for tokenzing the plain text
    # num_rets(optional) - default=30. the number of papers to extract.
    #
    # output: 
    # a list of most related papers. [(score, [paper_id, txt]) ... ]
    #################################################################

    print("Extracting similar papers ....")
    print("Query is :")
    print(query)
    paper_scores = []
    for paper in tqdm(data_dct):
        paper_id = paper['paper_id']
        candidate = paper['title'].strip()+". "+paper['abstract'].strip()
        score = text_similarity_score(query, candidate, model, tokenizer)
        paper_item = (score, [paper_id, candidate])
        paper_scores.append(paper_item)
    paper_scores.sort(key=lambda x: x[0], reverse=True)
    return paper_scores[:num_rets]


def extract_similar_txtsegs (query, data_dct, model, tokenizer, num_rets=60):
    #################################################################
    # Compute Extract most related text segments for the query
    # input: 
    # query - the query text for extraction
    # data_dct - data structure contians all paper information
    # model - similarity comparison model
    # tokenizer - tokenizer for tokenzing the plain text
    # num_rets(optional) - default=30. the number of segments to extract.
    #
    # output: 
    # a list of most related segments. [(score, [paper_id, txt]) ... ]
    #################################################################

    print("Extracting similar text segments ....")
    print("Query is :")
    print(query)
    seg_scores = []
    for paper in tqdm(data_dct):
        paper_id = paper['paper_id']
        candidates = paper['body_text']
        candidates = [item['text'].strip() for item in candidates]
        for candidate in candidates:
            score = text_similarity_score(query, candidate, model, tokenizer)
            seg_item = (score, [paper_id, candidate])
            seg_scores.append(seg_item)
    seg_scores.sort(key=lambda x: x[0], reverse=True)
    return seg_scores[:num_rets]


# **Example using these functions**
# * We use the json file for biorxiv track. 
# * In this example, I only use the "bert-base-uncased". You can use other pre-trained model or fine-tuning by yourself to get better results.

# **Define variables**

# In[ ]:


tokenizer_backbone = '/kaggle/input/pytorchbertbaseuncased/bert-base-uncased/bert-base-uncased-vocab.txt'
model_backbone = '/kaggle/input/pytorchbertbaseuncased/bert-base-uncased/'
full_docs_json = "/kaggle/working/full_docs_full_docs_biorxiv.json"
query = "The range of incubation periods for the disease in humans is important to learn."


# **Load Model**

# In[ ]:


print("Load Model ... ")
tokenizer = BertTokenizer.from_pretrained(tokenizer_backbone)
model = BertForNextSentencePrediction.from_pretrained(model_backbone)


# **Load Data**

# In[ ]:


print("Load Data ... ")
if full_docs_json is not None and full_docs_json!="":
    with open(full_docs_json, "r", encoding="utf8") as f:
        full_docs = json.load(f)
else:
    full_docs = create_docs_json("2020-03-13/biorxiv_medrxiv/biorxiv_medrxiv", save_file_name='full_docs_full_docs_biorxiv.json')


# **Extract Documents**
# * The runing time is a little slow. You can try by yourself or use mini-batch to speed up.

# In[ ]:


print("Extracting documents .... ")
if full_docs_json is not None and full_docs_json!="":
    extracted_papers = extract_similar_papers(query, full_docs, model, tokenizer)
    print(extracted_papers[0])


# **Extract Paragraphs/Segments**
# * The runing time is a little slow. You can try by yourself or use mini-batch to speed up.

# In[ ]:


print("Extracting paragraphs .... ")
# if full_docs_json is not None and full_docs_json!="":
    # extracted_segs = extract_similar_txtsegs(query, full_docs, model, tokenizer)
    # print(extracted_segs[10])

