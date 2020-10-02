#!/usr/bin/env python
# coding: utf-8

# ## COVID-19 Open Research Dataset Challenge (CORD-19)
# 
# An AI challenge to to the world's artificial intelligence experts to develop text and data mining tools that can help the medical community develop answers to high priority scientific questions. The CORD-19 dataset represents the most extensive machine-readable coronavirus literature collection available for data mining to date. This allows the worldwide AI research community the opportunity to apply text and data mining approaches to find answers to questions within, and connect insights across, this content in support of the ongoing COVID-19 response efforts worldwide. There is a growing urgency for these approaches because of the rapid increase in coronavirus literature, making it difficult for the medical community to keep up.

# ## Our approach
# 
# In this attempt of solving the above mentioned challenge, we use community creation technique for cluster creations for query expansion of the search. Most of this work is highly influenced by our paper - https://arxiv.org/pdf/2002.02238.pdf, which somewhat works on the same ground. 
# 
# The various steps of our method are as follows:
# 1. We include all the documents present in the dataset, remove duplicates present in the dataset based on 'SHA' values.
# 2. We consider the abstract of the papers retrieved and processed considering the above process. We only considered papers whose abstract was present, removed the rest.
# 3. We removed stopwords ,numerical values and punctuations from the processed dataset and then lemmatized the words for model training.
# 4. We train Word2Vec model on the dataset with the embedding size of 256. (The dimension of embedding was calculated from this work - https://github.com/ziyin-dl/word-embedding-dimensionality-selection)
# 5. Calculating the embedding size based on the dataset helped us to generate knowledge infused word embeddings tailored to domain knowledge.
# 6. After the embedding generation, we used the top 50000 words present in the word2vec model with their 10 most similar words for generating semantic clusters and capture contextual information of these words.
# 7. The above mentioned words were used to built a corpus graph which helps in finding similar communities present in the dataset and using this information for answering the queries of the task and challenge.
# 8. Then, we built a BM25 search index on the dataset with the keywords of the abstract paper infused by knowledge of corpus graph for Query Expansion methodology.
# 9. Lastly, we retrieve the results of the query of the task through two different techniques- Token based search and Embedding based search to demonstrate the efficacy of our methods.
# 
# 
# ## Enter your search query here
# 
# Enter the search query here.

# In[ ]:


search_query = 'anticoagulant therapy treatment hypercoagulable state thrombophilia anticoagulation'


# In[ ]:


get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz')
get_ipython().system('pip install en_core_sci_md')
get_ipython().system('pip install gensim')
get_ipython().system('pip install sklearn')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install networkx')
get_ipython().system('pip install wordcloud')
get_ipython().system('pip install rank-bm25')


# In[ ]:


# Importing relavant packages and setting packages
import numpy as np
import pandas as pd
import os
import json
import glob
import sys

from rank_bm25 import BM25Okapi
from networkx.algorithms.community import k_clique_communities

#search_query = "Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery."


# In[ ]:


# Getting all the files saved into a list and then iterate over them like below
# to extract relevant information

# Hold this information in a dataframe and then move forward from there.

# Just set up a quick blank dataframe to hold all these medical papers.

corona_features = {"doc_id": [None], "source": None, "title": [None], "abstract": [None],
                  "text_body": [None]}

corona_df = pd.DataFrame.from_dict(corona_features)


# In[ ]:


# accessing all the json files. Using glob for the job
root_dir = '/kaggle/input/CORD-19-research-challenge'
#json_filenames = glob.glob(f'{root_dir}/**/*.json', recursive=True)


# In[ ]:


# we read through the metadata file for finding duplicate papers and relevant data points for the search engine
sources = pd.read_csv(f'{root_dir}/metadata.csv')
sources.drop_duplicates(subset=['sha'], inplace=True)

def doi_url(d):
    if d.startswith('http://'):
        return d
    elif d.startswith('doi.org'):
        return f'http://{d}'
    else:
        return f'http://doi.org/{d}'
    
sources.doi = sources.doi.fillna('').apply(doi_url)


# In[ ]:


len(sources)


# In[ ]:


sources.head()


# In[ ]:


all_texts = sources.abstract
all_texts.dropna(inplace=True)


# In[ ]:


import gc
gc.collect()


# ### Text Preprocessing
# 
# Using spacy's biomedical tokenizer for text cleaning and basic preprocessing like stopwords removal, word in lemma form, etc.

# In[ ]:


# text preprocessing step
import en_core_sci_md
nlp = en_core_sci_md.load(disable=["tagger", "parser", "ner"])
nlp.max_length = 2000000


# In[ ]:


# Tokenizer function to generate tokens from the text file based on spacy's biomedical tokenizer
def spacy_tokenizer(sentence):
    # remove numbers (e.g. from references [1], etc.)
    try:
        return [word.lemma_ for word in nlp(sentence) if not (word.like_num or word.is_stop or word.is_punct or word.is_space)]
    except:
        return None


# In[ ]:


# removing customized stopwords prevalent in the domain of medicine research
customize_stop_words = ['doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 
    'et', 'al', 'author', 'figure', 'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'fig', 'fig.', 
    'al.', 'di', 'la', 'il', 'del', 'le', 'della', 'dei', 'delle', 'una', 'da',  'dell',  'non', 'si']

# Mark them as stop words
for w in customize_stop_words:
    nlp.vocab[w].is_stop = True


# In[ ]:


#saving merge dataframe
all_texts = all_texts.apply(spacy_tokenizer)


# ### Word2Vec Model Creation
# 
# Creating Word2Vec for the biomedical domain to understand relevant words and generating their embeddings.
# This is useful in generating a word cluster graph on the dataset and finding relevance between different words and clusters

# In[ ]:


from gensim.models import Word2Vec
from matplotlib import pyplot
import gensim

model = Word2Vec(all_texts, min_count=1, size=256, workers=4)

X = model[model.wv.vocab]
words = model.wv.vocab


# In[ ]:


len(words)


# In[ ]:


# Using a frequency distribution calculation to find relevant words present in the dataset to emphasis on important
# terms and find relationship between these words

word_freq = dict()
words = list(model.wv.vocab)
for i in range(len(words)):
    words[i] = words[i].replace("'","")
    if words[i] not in word_freq:
        if len(words[i]) > 2:
            word_freq[words[i]] = 0
        
print(len(list(word_freq.keys())))

no_list = all_texts.index
for i in no_list:
    for j in range(len(all_texts[i])):
        tmp = all_texts[i][j]
        if tmp in word_freq:
            word_freq[tmp] += 1 


# In[ ]:


word_freq = {k: v for k, v in sorted(word_freq.items(), key=lambda item: item[1], reverse=True)}


# ### Graph Generation
# 
# Graph is created to find relationship between words and finding various communities present in the dataset
# 
# Finding topn similar words for word in our vocabulary. This will help in recognizing clusters present in the data 

# In[ ]:


import networkx as nx
from scipy.spatial.distance import cosine
G2 = nx.Graph()


# In[ ]:


#G2 = nx.Graph()
words = list(word_freq.keys())
for i in range(len(words)):
       
    try:
        other_words = list(model.similar_by_word(words[i], topn=10))
    except:
        pass
    # print(other_words[0][0])
    
    if len(words[i]) > 2:
        G2.add_node(words[i])
        
        for j in range(len(other_words)):
            words[i] = words[i].replace("'","")
            w = other_words[j][0].replace("'","") 
            if len(w) > 2:
                G2.add_node(w)
                try:
                    sim_score = 1/(1 - cosine(model[words[i]], model[w]))
                    G2.add_weighted_edges_from([(words[i], w, sim_score)])
                except:
                    pass
        
print(G2.number_of_edges())


# In[ ]:


root_out_dir = '/kaggle/working'
nx.write_gpickle(G2, f"{root_out_dir}/graph_2_256.pkl")


# In[ ]:


G2.number_of_nodes()


# In[ ]:


G2.number_of_edges()


# In[ ]:


# Writing communities created in a file for better evaluation of the algorithm on the dataset
def comm_hyper(G2,k):
    c = list(k_clique_communities(G2, k))
    print(len(c))
    with open("set_"+str(k)+".txt","w", encoding='utf8') as f:
        for i in range(len(c)):
            f.write("\n------------------------------------------------------\n")
            f.write(str(len(c[i]))+"\n")
            f.write(str(c[i]))


# ### Community Identification
# 
# The important point is to look at the communities generated and choose the best set of generation among that. Too big a community with various informational sections isn't preferred so we need to narrow such communities down with the help of hyperparameter 'k'.

# In[ ]:


# Performing Clique Perlocation Technique for community finding in the graph created above.
# This step is more like tuning step where the value of "k" can be changed for various reasons
# comm_hyper(graph, k)

comm_hyper(G2, 4)
comm_hyper(G2, 5)
comm_hyper(G2, 6)


# In[ ]:


comm_4 = list(k_clique_communities(G2, 5))


# In[ ]:


word_1 = list(comm_4[14])


# In[ ]:


# Word Cloud creation for visualization 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

cloud_word = ' '
for word in word_1:
    cloud_word = cloud_word + word + ' '
    
stopwords = set(STOPWORDS)
wordcloud = WordCloud(width=800, height=800, background_color='white',
                     stopwords=stopwords, min_font_size=10).generate(cloud_word)

# plotting the word cloud image
plt.figure(figsize=(8,8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()


# ## Community Analysis

# In[ ]:


# Statistical analysis for communities based on number of words and different overlapping topics

max_len = 0
avg_len = 0
min_len = len(comm_4[0])

for i in range(len(comm_4)):
    temp_len = len(comm_4[i])
    if temp_len > max_len:
        max_len = temp_len
        
    if temp_len < min_len:
        min_len = temp_len
        
    avg_len += temp_len
    
print(max_len)
print(min_len)
print(avg_len/(len(comm_4)))


# ### Community Selection
# 
# We looked at all the communities and figured out the fact that communities with less members are mostly insignificant to major knowledge discovery and analysis, so we removed communities with length less than the average length of communities. This step helps us in keeping the information boom in check for retrieval purposes

# In[ ]:


comm = dict()
for i in range(len(comm_4)):
    if len(comm_4[i]) > 10:
        comm[i] = list()
        word_list = comm_4[i]
        for word in word_list:
            comm[i].append(word.lower())


# In[ ]:


print(len(list(comm.keys())))


# In[ ]:


columns = list()
columns.append("doc_id")
for cid in comm.keys():
    columns.append(cid)
comm_data = pd.DataFrame(columns=columns)


# ### Text Retrieval Part
# 
# Performing query expansion with community matching using both embedding techniques and token based matching

# In[ ]:


#search_query = search_query +" therapeutics, interventions and clinical studies. hypercoagulable state in COVID-19. efficacy of novel therapeutics anticoagulant therapy treatment hypercoagulable state thrombophilia anticoagulation"

# matching query with communities - token search
comm_list = dict()
check = spacy_tokenizer(search_query)
max_key = list(comm.keys())[0]
for key in comm.keys():
    freq = [1 if word in check else 0 for word in comm[key]]
    comm_list[key] = sum(freq)
    if comm_list[key] > comm_list[max_key]:
        max_key = key
    
print(comm_list[max_key])
print(max_key)


# In[ ]:


def get_word_frequency(word, text):
    count = 0
    for wd in text:
        if wd == word:
            count += 1
            
    return count


# In[ ]:


# using smooth inverse frequency for calculation of sentence embedding
def query_vector(query, embedding_size=256, a=1e-3):
    vs = np.zeros(embedding_size)
    for word in query:
        a_value = a /(a + get_word_frequency(word, query))
        vs = np.add(vs, np.multiply(a_value, model[word]))
        
    vs = np.divide(vs, len(query))
    return vs


# In[ ]:


# consolidated sentence embedding from word2vec
vs = query_vector(check)


# In[ ]:


comm_vec = dict()
for key in comm.keys():
    try:
        temp = query_vector(comm[key])
        comm_vec[key] = temp
    except:
        pass


# In[ ]:


from scipy.spatial.distance import cosine
max_score = 0
for key in comm_vec.keys():
    temp_score = 1/(1-cosine(vs, comm_vec[key]))
    if temp_score > max_score:
        max_score = temp_score
        key_val = key


# In[ ]:


def embedding_community_matching(query, comm, key_val):
    # embedding based community search scope expansion
    for word in comm[key_val]:
        query.append(word)
        
    return query


# In[ ]:


def community_matching(query, comm):
    # token based community search
    temp = 0
    for key in comm.keys():
        res = 0
        for word in query:
            if word in comm[key]:
                res += 1
        if res >= temp:
            temp = res
            comm_key = key
            
    for word in comm[comm_key]:
        query.append(word)
    return query


# ## BM25 Search Engine
# 
# For indexing the documents and searching the query based on query expansion from community generation to retreive relevant documents

# In[ ]:


bm25_index = BM25Okapi(all_texts.tolist())


# In[ ]:


def search(search_tokens, num_results=10):
    scores = bm25_index.get_scores(search_tokens)
    top_indexes = np.argsort(scores)[::-1][:num_results]
    
    return top_indexes,scores

# text = "novel corona virus"
# indexes = search(spacy_tokenizer(text))
# indexes


# ### Token Based Search with Community Information

# In[ ]:


query_text = spacy_tokenizer(search_query)
query_text = community_matching(query_text, comm)


# In[ ]:


indexes, scores = search(query_text, 50)


# In[ ]:


sources.loc[sources.index[indexes]]


# ### Embedding based Search with Community Information

# In[ ]:


query_text = spacy_tokenizer(search_query)
query_text = embedding_community_matching(query_text, comm, key_val)


# In[ ]:


indexes2, scores = search(query_text, 50)


# In[ ]:


sources.loc[sources.index[indexes2]]


# ## Finding Query Words in Papers

# In[ ]:


import html
import random
from IPython.core.display import display, HTML

# Prevent special characters like & and < to cause the browser to display something other than what you intended.
def html_escape(text_body):
    return html.escape(text_body)


def text_highlight(text_body, query_text):
    # Remove duplicate words from text_body
    seen = set()

    for item in text_body:
        if item not in seen:
            seen.add(item)

    # Create random sample weights for each unique word
    weights = []
    for i in range(len(query_text)):
        weights.append(0.5)

    df_coeff = pd.DataFrame({'word': query_text, 'num_code': weights})

    # Select the code value to generate different weights
    word_to_coeff_mapping = {}

    for row in df_coeff.iterrows():
        row = row[1]
        word_to_coeff_mapping[row[0]] = row[1]

    max_alpha = 0.8
    highlighted_text_body = []
    for word in text_body:
        try:
            weight = word_to_coeff_mapping[word]
        except:
            weight = None
        
        if weight is not None:
            highlighted_text_body.append('<span style="background-color:rgba(135,206,250,' + str(weight/max_alpha) + ');">' + html_escape(word) + '</span>')
        else:
            highlighted_text_body.append(word)

    highlighted_text_body = ' '.join(highlighted_text_body)
    display(HTML(highlighted_text_body))


# In[ ]:


text_highlight(all_texts.loc[sources.index[indexes2[2]]], query_text)


# In[ ]:


new_index = sources.index[indexes2]


# In[ ]:


#Now we iterate over the files and populate the data frame

def return_corona_df(json_filenames, df, sources, new_index):

# len(json_filenames)
    for i in range(len(json_filenames)):
        file_name = json_filenames[i]
        row = {"doc_id": None, "source": None, "title":None, "abstract":None, 
               "text_body":None}

        with open(file_name) as json_data:
            data = json.load(json_data)

            row['doc_id'] = data['paper_id']
            row['title'] = data['metadata']['title']

            # Now need all of abstract. Put it all in a list then use str.join()
            # to split it into paragraphs


            if sources.loc[new_index[i]]['pdf_json_files'] != 'NaN':
                row['abstract'] = sources.loc[new_index[i]]['abstract']

            # And lastly the body of the text to be added.

            body_list = []
            for _ in range(len(data['body_text'])):
                try:
                    body_list.append(data['body_text'][_]['text'])
                except:
                    pass

            body = "\n".join(body_list)
            row['text_body'] = body

            # Now just add to the dataframe
            df = df.append(row, ignore_index=True)
            del row

    return df


# In[ ]:


json_filenames = list()
for i in range(len(new_index)):
    if sources.loc[new_index[i]]['pdf_json_files'] != 'NaN':
        json_filenames.append(f'{root_dir}/'+sources.loc[new_index[i]]['pdf_json_files'].split(';')[0])


# In[ ]:


corona_df = return_corona_df(json_filenames, corona_df, sources, new_index)


# In[ ]:


corona_df = corona_df.iloc[1:].reset_index(drop=True)
corona_df.drop_duplicates(subset=['doc_id'], inplace=True)


# In[ ]:


corona_df.head()


# ### Drug Discovery

# In[ ]:


from bs4 import BeautifulSoup
import requests as r
import string
import time
import re

alphabet = list(string.ascii_lowercase)

links = []
for letter in alphabet:
    links.append(f'https://druginfo.nlm.nih.gov/drugportal/drug/names/{letter}')
    
pages = []
for link in links:
    page = r.get(link)
    pages.append(page.text)
    time.sleep(3)
    
drug_tables = []
for p in pages:    
    parser = BeautifulSoup(p, 'html.parser')
    tables = parser.find_all('table')
    drug_tables.append(tables[2])
    
drug_names = []
for table in drug_tables:
    a_refs = table.find_all('a')
    for a in a_refs:
        drug_names.append(a.string)
        
drug_names.append('oseltamivir') #adding other relevant drugs by hand
drug_names.append('lopinavir') #adding other relevant drugs by hand

nih_dnames = pd.DataFrame({'drug_name':drug_names})
nih_dnames.drug_name = nih_dnames.drug_name.str.lower()
nih_dnames.to_csv('nih_dnames.csv', index=False)


drug_names = nih_dnames.drug_name

def find_drugs(df):
    mentioned_drugs = []
    for article in df.iterrows():
        article_drugs = ''
        for drug in drug_names:
            if re.search(fr'([\W\b])({drug})([\b\W])', article[1]['text_body'].lower()) != None:
                article_drugs = article_drugs + drug + ';'
        if (len(article_drugs) > 0):
            article_drugs = article_drugs[:-1]
        mentioned_drugs.append(article_drugs)
    return mentioned_drugs


search_drugs = find_drugs(corona_df)
corona_df['mentioned_drugs'] = search_drugs


# In[ ]:


corona_df.head()


# In[ ]:


corona_df.head()


# In[ ]:


page = r.get('https://www.ef.edu/english-resources/english-grammar/numbers-english/')
parser = BeautifulSoup(page.content, 'html.parser')
tables = parser.find_all('table')
cardinal_numbers = pd.read_html(str(tables[0]))[0].loc[0:37, "Cardinal"]

def find_sample_sizes(df):
    patient_num_list = []
    for abstract in df.text_body:
        matches = re.findall(r'(\s)([0-9,]+)(\s|\s[^0-9\s]+\s)(patients)', abstract)
        num_patients = ''
        for match in matches:
            num_patients = num_patients + ''.join(match[1:]) + ';'
        for number in cardinal_numbers:
            cardinal_regex_search = re.search(fr'({number})(\s)(patients)', abstract)
            if cardinal_regex_search != None:
                num_patients = num_patients + cardinal_regex_search[0] + ';'
        num_patients = num_patients[:-1]
        patient_num_list.append(num_patients)
    return patient_num_list

corona_df['sample_size'] = find_sample_sizes(corona_df)
corona_df.head()


# In[ ]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def remove_stopwords(words):
    stop_words = stopwords.words('english')
    clean = [word for word in words if word not in stop_words]
    return clean

study_type_adjs = ['cohort', 'observational', 'clinical', 
              'randomized', 'open-label', 'control',
              'case', 'meta-analysis', 'systematic',
              'narrative', 'literature', 'critical',
              'retrospective', 'controlled', 'non-randomized',
              'secondary', 'rapid', 'double-blinded',
              'open-labelled', 'concurrent', 'pilot', 
              'empirical', 'retrospective', 'single-center',
              'collaborative', 'case-control']
study_type_nouns = ['study', 'trial', 'report', 'review',
                   'analysis', 'studies']

def clean(string):
    temp = re.sub(r'[^\w\s]', '', string)
    temp = re.sub(r'\b\d+\b', '', temp)
    temp = re.sub(r'\s+', ' ', temp)
    temp = re.sub(r'^\s', '', temp)
    temp = re.sub(r'\s$', '', temp)
    return temp.lower()

def find_study_types(df):
    study_types = []
    for article in df.iterrows():
        title = article[1].title.lower()
        clean_title = clean(title)
        title_tokens = word_tokenize(clean_title)
        title_tokens = remove_stopwords(title_tokens)
        study_type = ''
        study_type_name = False

        article_study_type = []
        for word in title_tokens:
            if word in study_type_adjs:
                study_type = study_type + word + ' '
                study_type_name = True

            elif word in study_type_nouns:
                study_type = study_type + word + ' '
                study_type_name = True

            if (study_type_name == False):
                study_type = ''

        study_types.append(study_type[:-1])

    return study_types

corona_stypes = find_study_types(corona_df)
corona_df['study_types'] = corona_stypes
corona_df.head()


# In[ ]:


severity_types = ['severe', 'critical', 'icu', 'mild']

def find_severities(df):
    severity_of_disease = []
    for abstract in df.text_body:
        text = re.sub('Severe acute respiratory', 'sar', abstract)
        severities = []
        tokens = word_tokenize(text)
        for severity_type in severity_types:
            if severity_type in tokens:
                severities.append(severity_type)
        severity_of_disease.append(';'.join(severities))
    return severity_of_disease

corona_severities = find_severities(corona_df)
corona_df['severity_of_disease'] = corona_severities
corona_df.head()


# In[ ]:


page = r.get('https://www.ef.edu/english-resources/english-grammar/numbers-english/')
parser = BeautifulSoup(page.content, 'html.parser')
tables = parser.find_all('table')
cardinal_numbers = pd.read_html(str(tables[0]))[0].loc[0:37, "Cardinal"]

def find_sample_sizes(df):
    patient_num_list = []
    for abstract in df.text_body:
        matches = re.findall(r'(\s)([0-9,]+)(\s|\s[^0-9\s]+\s)(patients)', abstract)
        num_patients = ''
        for match in matches:
            num_patients = num_patients + ''.join(match[1:]) + ';'
        for number in cardinal_numbers:
            cardinal_regex_search = re.search(fr'({number})(\s)(patients)', abstract)
            if cardinal_regex_search != None:
                num_patients = num_patients + cardinal_regex_search[0] + ';'
        num_patients = num_patients[:-1]
        patient_num_list.append(num_patients)
    return patient_num_list

corona_df['sample_size'] = find_sample_sizes(corona_df)
corona_df.head()


# In[ ]:


def output_csv(df, new_index):
    
    i = 0
    results = pd.DataFrame(columns=['cord_id','date','study','study_link','journal', 
                        'study_type', 'therapeutic_method', 'sample_size', 
                        'severity_of_disease', 'conclusion_excerpt', 'primary_endpoints', 
                        'clinical_imporovement', 'added_on'])
    for j in df.index:
        #print(paper_row)
        new_row = {'cord_id': sources.loc[new_index[i]]['cord_uid'],
                   'date':sources.loc[new_index[i]]['publish_time'],
                   'study_link':[sources.loc[new_index[i]]['url']], 
                    'journal':[sources.loc[new_index[i]]['journal']], 
                   'study_type':[df.loc[j].study_types], 
                   'therapeutic_method':[df.loc[j].mentioned_drugs], 
                    'sample_size':[df.loc[j].sample_size], 
                   'severity_of_disease':[df.loc[j].severity_of_disease], 
                   'conclusion_excerpt':[df.loc[j].text_body], 
                    'primary_endpoints':[''], 
                   'clinical_improvement':[''], 
                   'added_on':['']}
        
        results = results.append(new_row, ignore_index=True)
        i = i+1
        
    return results


# In[ ]:


results = output_csv(corona_df, new_index)


# In[ ]:


results.head()


# In[ ]:


results.to_csv('results.csv', index=False)


# In[ ]:




