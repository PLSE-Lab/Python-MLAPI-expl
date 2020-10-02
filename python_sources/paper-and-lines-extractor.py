#!/usr/bin/env python
# coding: utf-8

# ## Aim of the Notebook 
# 
# * Given a Query, Search for related papers. 
# * Papers are searched using their Title and Abstract 
# * Once potential papers are searched, find specific lines in the Papers talking about the given query.
# * Using BM25 Algorithm, used for creating search engines and also to search lines in Potential papers.  
#   Read about the algorithm: [https://en.wikipedia.org/wiki/Okapi_BM25]
# * FINAL OUTCOME: Find the right papers for a given query, and then highlight/find right lines in the paper.
# * Discliamer: Data Analysis / Visualization is not done extensively. THE FOCUS IS ON CREATING ALGORITHM WHICH AND FIND RIGHT LINES IN THE PAPERS DEALING WITH THE GIVEN QUERY.
# 
#     

# ### Download the Dependencies

# In[ ]:


get_ipython().system('pip install rank_bm25 nltk')
get_ipython().system('pip install scispacy')
get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz')


# ### Import the dependencies

# In[ ]:


import numpy as np # linear algebra
import os
import json
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import re
import string
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import spacy
import  scispacy
from nltk.tokenize import sent_tokenize
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd


# ### Load All Papers (Jsons) in a Dataframe

# In[ ]:


df = pd.DataFrame(columns=['paper_id', 'title', 'abstract', 'body_text'])
i = 0
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.endswith('.json')==True:
            dic = json.loads(open(os.path.join(dirname, filename), 'r').read())
            id  = dic['paper_id']
            title = dic['metadata']['title']
            if title.strip()=='':
                title = None
            abstract = dic['abstract']
            abstract_list = []
            for a in abstract:
                t = a['text']
                abstract_list.append(t)
            abstract_string = '\n'.join(abstract_list)
            
            if abstract_string == '':
                abstract_string = None

            body_text = dic['body_text']
            body_texts_list = []
            for text in body_text:
                t = text['text']
                body_texts_list.append(t)
                
            body_text_string = '\n'.join(body_texts_list)
            body_text_string = body_text_string.strip()
            if body_text_string == '':
                body_text_string = None
                
            df.loc[i] = [id, title, abstract_string, body_text_string]
            i+=1
                
            


# In[ ]:


## Set the paper id as the index
df = df.set_index('paper_id')


# In[ ]:


print (df.info())


# In[ ]:


df.describe()


# In[ ]:


### Remove the null values
df.dropna(inplace=True)


# In[ ]:


df.info()


# In[ ]:


### Create a copy of the dataframe and work on that
df_work = df.copy()


# ### Preprocessing 

# In[ ]:


# medium model
import en_core_sci_md
nlp = en_core_sci_md.load(disable=["tagger", "parser", "ner"])
nlp.max_length = 2000000


# In[ ]:


# New stop words list 
customize_stop_words = [
    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 
    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'fig', 'fig.', 'al.',
    'di', 'la', 'il', 'del', 'le', 'della', 'dei', 'delle', 'una', 'da',  'dell',  'non', 'si'
]

# Mark them as stop words
for w in customize_stop_words:
    nlp.vocab[w].is_stop = True


# In[ ]:


stop_words = nlp.Defaults.stop_words


# In[ ]:


def spacy_tokenizer(sentence):
    return [word.lemma_ for word in nlp(sentence) if not (word.like_num or word.is_stop or word.is_punct or word.is_space or len(word)==1)]


# In[ ]:


s = ' i am a good-boy , i (am rishav) ?'
spacy_tokenizer(s)


# In[ ]:


df_work['body_text'] = df_work['body_text'].apply(spacy_tokenizer)
df_work['abstract'] = df_work['abstract'].apply(spacy_tokenizer)
df_work['title'] = df_work['title'].apply(spacy_tokenizer)


# In[ ]:


df_work['body_text'] = df_work['body_text'].apply(lambda x: ' '.join(x))
df_work['abstract'] = df_work['abstract'].apply(lambda x: ' '.join(x))
df_work['title'] = df_work['title'].apply(lambda x: ' '.join(x))


# In[ ]:


df_work.head()


# ### Paper to index mapping

# In[ ]:


paper_ids = df_work.index
id_2_paper = {}
for i, id in enumerate(paper_ids):
    id_2_paper[i] = id


# ### Let's take title and abstract for paper search

# In[ ]:


df_work['append_title_abstract'] = df_work['title'] + ' ' + df_work['abstract'] 


# ### Create a index of the Papers

# In[ ]:


corpus = df_work['append_title_abstract'].tolist()
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)


# ### Potential papers which could contain what we want

# In[ ]:


from IPython.display import display, Markdown


# In[ ]:


def markdown_print(line):
    print(line)


# In[ ]:


def suggestPapers(query):
    top_20_papers = []
    query_pre = spacy_tokenizer(query)
    doc_scores = bm25.get_scores(query_pre)
    doc_scores = [(score, ind) for ind, score in enumerate(doc_scores)]
    doc_scores = sorted(doc_scores, reverse = True)
    doc_score_top_20 = doc_scores[:20]
    for d in doc_score_top_20:
        i = d[1]
        top_20_papers.append((paper_ids[i], d[0]))
        
    suggested_papers_dict = {}
    for ele in top_20_papers:
        suggested_papers_dict[ele[0]] = ele[1]
        
    for paper in top_20_papers[:5]:
        markdown_print ("Paper Id: " +paper[0])
        markdown_print("Title: "+df.loc[paper[0]]['title'])
        markdown_print ("-----")
    return top_20_papers, suggested_papers_dict


# In[ ]:


def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stop_words,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


# ### Let's find the lines in the Paper talking about the given Query
# * Here, we are picking all the top 20 papers and forming a coupus of all pair from them. 
# * This will be our corpus where we will be searching our query on.

# In[ ]:


def create_paper_lines(suggested_papers):
    paper_lines = []
    for paper in suggested_papers:
        text = df.loc[paper[0]]['abstract']
        sentences = [(sent.strip(), paper[0]) for sent in sent_tokenize(text)]
        sentences_2_s = []
        for i in range(0, len(sentences)-1):
            sent1 = sentences[i][0]
            sent2 = sentences[i+1][0]
            sent = sent1 + ' ' + sent2
            sentences_2_s.append((sent, sentences[i][1]))
        paper_lines.extend(sentences_2_s)
    return paper_lines


# In[ ]:


def create_wordcloud(paper_lines):
    word_cloud_data = []
    for line, id in paper_lines:
        word_cloud_data.append(line)
    show_wordcloud(word_cloud_data)


# In[ ]:


def create_mapping(paper_lines):
    id_paper_line_paper_id = {}
    for i, tup in enumerate(paper_lines):
        id_paper_line_paper_id[i] = (tup[0], tup[1])
    return id_paper_line_paper_id

def create_paper_lines_index(paper_lines):
    paper_lines_fmtd = []
    for line in paper_lines:
        paper_lines_fmtd.append(" ".join(spacy_tokenizer(line[0])))
    tokenized_paper_lines = [doc.split(" ") for doc in paper_lines_fmtd]
    bm25_task1 = BM25Okapi(tokenized_paper_lines)
    return bm25_task1


# In[ ]:


def top_lines_suggestor(task, bm25_task1, id_paper_line_paper_id):
    top_50_lines = []
    task1_pre = spacy_tokenizer(task)
    line_scores = bm25_task1.get_scores(task1_pre)
    line_scores = [(score, ind) for ind, score in enumerate(line_scores)]
    line_scores = sorted(line_scores, reverse = True)
    line_scores_top_50 = line_scores[:50]
    for l in line_scores_top_50:
        i = l[1]
        paper_id = id_paper_line_paper_id[i][1]
        actual_line = id_paper_line_paper_id[i][0]
        line_score = l[0]
        paper_score = suggested_papers_dict[paper_id]
        net_score = paper_score * line_score
        top_50_lines.append((actual_line, net_score, paper_id))
    top_ten_sorted = sorted(top_50_lines, key = lambda x: x[1], reverse=True)[:20]
    for line in top_ten_sorted:
        markdown_print ("Paper line: " +line[0])
        markdown_print ("Paper id: " +  line[2])
        markdown_print ("Similarity Score: " + str(line[1]))
        markdown_print ("-----")


# ### Analysis of Incubation periods and Recovery time

# In[ ]:


task1 = 'Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious,even after recovery.'
suggested_papers, suggested_papers_dict = suggestPapers(task1)  ### TOP SUGGESTED PAPERS


# In[ ]:


paper_lines = create_paper_lines(suggested_papers)
show_wordcloud(paper_lines)  ### Show the wordcloud for the given topic, what words are prevlant


# In[ ]:


id_paper_line_paper_id = create_mapping(paper_lines)  
bm25_task = create_paper_lines_index(paper_lines)
top_lines_suggestor(task1, bm25_task, id_paper_line_paper_id) ### Print the top lines in the potential papers


# ### Prevalence of asymptomatic shedding and transmission (e.g., particularly children).**

# In[ ]:


task2 = 'Prevalence of asymptomatic shedding and transmission (e.g., particularly children).'
suggested_papers, suggested_papers_dict = suggestPapers(task2)  ### TOP SUGGESTED PAPERS


# In[ ]:


paper_lines = create_paper_lines(suggested_papers)
show_wordcloud(paper_lines)  ### Show the wordcloud for the given topic, what words are prevlant


# In[ ]:


id_paper_line_paper_id = create_mapping(paper_lines)  
bm25_task = create_paper_lines_index(paper_lines)
top_lines_suggestor(task1, bm25_task, id_paper_line_paper_id) ### Print the top lines in the potential papers


# ### Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).
# 

# In[ ]:


task3 = 'Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).'
suggested_papers, suggested_papers_dict = suggestPapers(task3)  ### TOP SUGGESTED PAPERS


# In[ ]:


paper_lines = create_paper_lines(suggested_papers)
show_wordcloud(paper_lines)  ### Show the wordcloud for the given topic, what words are prevlant


# In[ ]:


id_paper_line_paper_id = create_mapping(paper_lines)  
bm25_task = create_paper_lines_index(paper_lines)
top_lines_suggestor(task3, bm25_task, id_paper_line_paper_id) ### Print the top lines in the potential papers


# ### Immune response and immunity

# In[ ]:


task4 = 'Immune response and immunity'
suggested_papers, suggested_papers_dict = suggestPapers(task4)  ### TOP SUGGESTED PAPER4


# In[ ]:


paper_lines = create_paper_lines(suggested_papers)
show_wordcloud(paper_lines)  ### Show the wordcloud for the given topic, what words are prevlant


# In[ ]:


id_paper_line_paper_id = create_mapping(paper_lines)  
bm25_task = create_paper_lines_index(paper_lines)
top_lines_suggestor(task4, bm25_task, id_paper_line_paper_id) ### Print the top lines in the potential papers


# ### Natural history of the virus and shedding of it from an infected person**

# In[ ]:


task5 = 'Natural history of the virus and shedding of it from an infected person'
suggested_papers, suggested_papers_dict = suggestPapers(task5)  ### TOP SUGGESTED PAPER4


# In[ ]:


paper_lines = create_paper_lines(suggested_papers)
show_wordcloud(paper_lines)  ### Show the wordcloud for the given topic, what words are prevlant


# In[ ]:


id_paper_line_paper_id = create_mapping(paper_lines)  
bm25_task = create_paper_lines_index(paper_lines)
top_lines_suggestor(task5, bm25_task, id_paper_line_paper_id) ### Print the top lines in the potential papers


# In[ ]:




