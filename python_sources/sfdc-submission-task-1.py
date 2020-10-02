#!/usr/bin/env python
# coding: utf-8

# # What is known about transmission, incubation, and environmental stability?
# 

# # Task Description
# 
# Given the increasing number of scientific publications on COVID-19 and related topics, a semantic search engine capable of extracting relevant publications for complex questions and queries has the potential to decrease the difficulty of finding the right information in a timely and up-to-date fashion. Here we present a system architecture for semantic search across the provided COVID-19 publications, and show the output for some of the task-specific subqueries provided. 
# 
# For brevity, we do not include all of our code, given that it is designed to run on a web application. More below.

# # Search Engine
# 
# The running application can be found at https://sfr-med.com/search
# 
# ![search_engine_front_end.png](attachment:search_engine_front_end.png)

# ## Search Engine Architecture
# 
# The system is composed of two steps, below, along with a series of sub-modules.
# 
# ### Step 1. Document Retrieval
# First, we parse and vectorize the existing scientific literature into a document index, and retrieve documents from this index given a text query from the user. 
# 
# We  take the given Kaggle text corpus - the scientific articles, their paragraphs, image captions, and other components - and feed each into a TF-IDF vectorizer, which stores the resultant vectors into a document index. 
# 
# When a text query comes in from a user, we then apply the same TF-IDF vectorizer on the query to obtain a query vector. Using k-nearest-neighbors retrieval, we return the top-N similar *paragraph* vectors from the document index. We then rank these based on relevance, citation count, and recency to obtain a sorted list of the N most relevant paragraphs (and their corresponding documents) to a given query.
# 
# ![image.png](attachment:image.png)
# 
# 

# 
# ### Step 2: Question-answering 
# 
# Next, we take the retrieved documents, along with the text query, and run them through a question answering engine which returns a set of text answers. The system's final output is then the retrieved paragraphs with the text answers potentially highlighted, along with the document titles.
# 
# ![image.png](attachment:image.png)

# ## System Components
# In the subsections below we describe the key AI components of the system - the data puller, retrieval component, question answering component, and paragraph ranker.
# 
# 
# ### Data Puller
# The code in this section defines the classes needed for data pulling

# In[ ]:


import os
import logging
import requests
import numpy as np
import pandas as pd
from requests.exceptions import HTTPError, ConnectionError
import json

logging.getLogger()
logging.basicConfig(level=logging.DEBUG)

'''
Generate paras, image text, image/text OCR text
'''
class DataPuller():
    def __init__(self, input_dir, metadata_path):
        self.input_dir = input_dir
        self.metadata_path = metadata_path
        self.paper_details_df = None

    def load(self):
        '''
        Loads all the data and
        flattens the data structure by joining metadata with paragraphs
        Drops papers without full text. # TODO
        '''        
        metadata = pd.read_csv(self.metadata_path)
        
        # read paper details
        paper_texts = []
        for (dirpath, dirnames, filenames) in os.walk(self.input_dir):
            for file in filenames:
                if(file.endswith(".json")):
                    with open(os.path.join(dirpath, file), 'r') as f:
                        paper_texts.append(json.loads(f.read()))
        
        # merge paper details with metadata
        paper_details_df = pd.DataFrame.from_records(paper_texts)
        paper_details_df = paper_details_df.drop(labels=['abstract'], axis=1)
        logging.info("Loaded paper details for {} papers".format(len(paper_details_df)))
        metadata['paper_id'] = metadata['sha']
        paper_details_df = paper_details_df.merge(metadata, on='paper_id', how='left')
        paper_details_df = paper_details_df[paper_details_df['title'].notna()]
        self.paper_details_df = paper_details_df.reset_index()
        logging.info("Successfully merged paper details with metadata . Total records after merging {}".format(len(self.paper_details_df)))

        records = []
        count = 0  
        # Get all abstract first
        for i in metadata.index:            
            row = metadata.iloc[i]
            records.append(self.get_record(row=row, 
                                        idx=count, 
                                        doc=row['abstract'],
                                        doc_type='abstract'))
        
            count += 1
        logging.info("Loaded abstract for {} papers".format(count))
        for i in self.paper_details_df.index:            
            row = self.paper_details_df.iloc[i]
            
            for para in row['body_text']:
                records.append(self.get_record(row=row, 
                                                idx=count, 
                                                doc=para['text'],
                                                doc_type='para'))
                count += 1    

            for k, fig in row['ref_entries'].items():
                records.append(self.get_record(row=row, 
                                                idx=count, 
                                                doc=fig['text'],
                                                doc_type='caption'))
                count += 1

        docs_df = pd.DataFrame.from_records(records)
        logging.info("Data puller total records : {}".format(len(docs_df)))
        docs_df = docs_df.drop_duplicates(subset='id')
        logging.info("Data puller total records after dropping dupes : {}".format(len(docs_df)))
        return docs_df

    def get_record(self, row, idx, doc, doc_type):
        return {
            'id': str(row['paper_id']) + '_' + str(idx),
            'paper_id': row['paper_id'],
            'paper_title': row['title'],
            'display_text': doc,
            'text': str(row['title']) + ' . ' + str(doc),
            'doc_type': doc_type,
            'link': row['url'],
            'date': row['publish_time'],
            'authors': row['authors'],
            'journal': row['journal']
        }


# **Example of retrieved data in json format**

# In[ ]:


# Format:
# {
# 'id': <str>,
# 'paper_id': <str>,
# 'paper_title': <str>,
# 'display_text': <str>,
# 'text': <str>,
# 'doc_type': <str>, #['abstract', 'para', 'caption']
# 'link': <str>,
# 'date': <str>,
# 'authors': <str>,
# 'journal': <str>
# }


# ### TF-IDF & Retrieval
# Our text retrieval system and QA engine follows our recent work on open-domain QA over Wikipedia [(Asai et al., 2020)](https://arxiv.org/abs/1911.10470). Like Wikipedia articles, we first split all the biomedical research papers in our database into abstract and paragraphs, and build a TF-IDF retrieval model following [DrQA](https://github.com/facebookresearch/DrQA). We have two options for the term-based retrieval step, either direct paragraph-level retrieval or hierarchical retrieval (first, article-level retrieval, and then paragraph-level re-ranking, as in [Asai et al. (2020)](https://arxiv.org/abs/1911.10470)). We employed the former option as our first version. By either of the options, the term-based retrieval system returns top-N paragraphs, given an input query. The term-based retrieval system can be replaced with other search engines on demand.
# 
# ### Question-Answering and Paragraph Ranking
# Our system then uses the sequential paragraph selector model proposed in [Asai et al. (2020)](https://arxiv.org/abs/1911.10470). The official code and trained models are available on [Github](https://github.com/AkariAsai/learning_to_retrieve_reasoning_paths). The objective of this step is to further verify which paragraphs are more relevant to provide answers for each query. Instead of separately giving a relevance score for each of the top-N paragraphs, the sequential paragraph selector model can select sets of a few paragraphs by modeling relationships between them (i.e. using so-called "multi-hop" reasoning).
# 
# In that paper, all the experiments were conducted on Wikipedia-oriented datasets (HotpotQA, SQuAD, Natural Questions), but our target documents come from biomedical research articles. (For more details about the datasets, please refer to the paper.) To fill the gap, we further fine-tuned the HotpotQA-based model with the [PubMedQA dataset](https://github.com/pubmedqa/pubmedqa). Each training example in the PubMedQA dataset can be considered as a set of a question and its related paragraphs, and we assume the related paragraphs are the ground-truth paragraphs to be selected for the question. To train the sequential paragraph selector model, we also need negative examples, and we follow [Asai et al. (2020)](https://arxiv.org/abs/1911.10470) to use paragraphs with the highest TF-IDF scores given the question. We modified the original beam search, so that different paths in the beam search can include more diverse paragraphs to avoid extracting the same answers from different paths.
# 
# Finally, the selected sets (or paths) of the paragraphs are fed into an extractive reading comprehension model based on [Asai et al. (2020)](https://arxiv.org/abs/1911.10470). There are three models for the three datasets, HotpotQA, SQuAD, and Natural Questions, and we observed that the SQuAD-based model matches our demand for the COVID-19 project, based on the question types covered by the different datasets. (It is worth trying to further fine-tune the models with biomedical extractive QA datasets.)
# 
# Unlike working on the benchmark QA datasets, there are no ground-truth answers in the real-world application, and we do not need to output the best hypothesis from the model. Therefore our system highlights multiple answer candidates (text spans) extracted from the selected paragraphs, so that the users can decide which is more useful.

# # Results
# 
# In the cells below, we sequentially run a few of the provided task queries (below) through our system.
# 
# **Task Details**<br>
# What is known about transmission, incubation, and environmental stability? What do we know about natural history, transmission, and diagnostics for the virus? What have we learned about infection prevention and control?
# 
# **System queries:**
#  - What is the range of incubation periods for COVID-19 in humans?
#  - How long are individuals contagious?
#  - What is the prevalence of asymptomatic shedding and transmission?
#  - What is the prevalence of asymptomatic shedding and transmission in children?
#  - Seasonality of transmission.
#  - Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).
#  - Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).
#  - Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).
#  - Natural history of the virus and shedding of it from an infected person
#  - Implementation of diagnostics and products to improve clinical processes
#  - Disease models, including animal models for infection, disease and transmission
#  - Tools and studies to monitor phenotypic change and potential adaptation of the virus
#  - Immune response and immunity
#  - Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings
#  - Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings
#  - Role of the environment in transmission
# 

# In[ ]:


from urllib.request import urlopen
from IPython.display import IFrame
from urllib.parse import quote

def query_to_url(query):
    return "https://sfr-med.com/search?q=" + quote(query + " COVID -19")


def render_url(url):
    html = urlopen(url).read()
    html = html.decode("utf-8")

    # Eliminate header and search bar
    html = html.split("<div class=\"slds-col slds-large-size_2-of-12\">")[0] + html.split("</form>")[1]
    
    filename = "./tmp.html"
    open(filename, "w").write(html)
    return IFrame(src=filename, width=1300, height=600)


def display_query(query):
    try:
        url = query_to_url(query)
        display(render_url(url))
    except:
        print("Error rendering...")

        
queries = """
 - What is the range of incubation periods for COVID-19 in humans?
 - What is the prevalence of asymptomatic shedding and transmission?
 - What is the prevalence of asymptomatic shedding and transmission in children?
 - Seasonality of transmission.
 - How long are individuals contagious?
 - Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).
 - Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).
 - Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).
 - Natural history of the virus and shedding of it from an infected person
 - Implementation of diagnostics and products to improve clinical processes
 - Disease models, including animal models for infection, disease and transmission
 - Tools and studies to monitor phenotypic change and potential adaptation of the virus
 - Immune response and immunity
 - Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings
 - Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings
 - Role of the environment in transmission
""".strip().split("\n")


# In[ ]:


import time
for query in queries[:1]:
    query = query.strip()
    print(query)
    display_query(query)
    time.sleep(0.5)


# ## Authors (in no particular order): 
#  - Andre Esteva, AndreEstevaSFDC, aesteva@salesforce.com
#  - Anuprit Kale, AnupritK, akale@salesforce.com
#  - Dragomir Radev, dragomirradev, dradev@salesforce.com
#  - Kazuma Hashimoto, kazumahashimoto, k.hashimoto@salesforce.com
#  - Wenpeng Yin, wyin@salesforce.com
#  - Romain Paulus, romainfpaulus, rpaulus@salesforce.com
#  - Richard Socher, richards, rsocher@salesforce.com

# In[ ]:




