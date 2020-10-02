#!/usr/bin/env python
# coding: utf-8

# # Basic idea of the kernel
#  
# We joined the competition to contribute an advanced search engine help search for research papers. The idea was to develop a tool that is able to search relevant for a given word vector, very much alike a traditional search engine tool. The paper is then ranked in relevance to the established word vector. Afterwards our tool generates a heatmap cluster for the N most relevant papers, helping to identify similar papers. The goal for generating this heatmap was to quickly gain insight into a specific topic and find corresponding papers, so that a comprehensive insight is possible.
# 
# Ultimately, the tool can then produce a summary for each of the papers either using a SIF-like, TF-IDF based approach, or genomic summary function. In our tests, the generic gensim summary function generated the most general, but also most easy to read summaries. If a comprehensive insight was needed, the TF-IDF approach performed the best. It allows to give priority to certain keywords, so that the reader can quickly identify important statements on specific sub-topics. With this research tool we hope to help researchers navigate more easily through the ever-growing number of papers regarding the SARS-COVID 19 pandemic so that they may find adequate countermeasures to combat the virus.
# 
# In the second submission period we will focus our efforts on improving the usability of the kernel through a more sophisticated visualisation, and an improved search algorithm. We believe PageRank can be used to find the most relevant papers and would be able to improve the quality of searches but are concerned that papers that are still in the process of being published could screw the results.
# 
# For now, we are happy to showcase our engine and guide you through our implementation process step-by-step on the next section. If you are only interested in using the tool, you can use the following code block to operate the system. (Please make sure to initialize all functions first and import all necessary libraries)
# 

# In[ ]:


import numpy as np 
import pandas as pd
import json
import matplotlib.pyplot as plt
import glob
import os
import math
import csv
import gc
import re

from multiprocessing import Process, Manager

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer 

from sklearn.metrics.pairwise import cosine_similarity

from ast import literal_eval

from gensim.models import Word2Vec

plt.style.use('ggplot')
root_path = '/kaggle/input/CORD-19-research-challenge/'
#metafile = open('../input/CORD-19-research-challenge/metadata.csv')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

w2v = Word2Vec.load('../input/models/word2vec/w2v.model')


# # Preprocessing of the Data
# 
# Before we started working with the given data we went through the common steps of natural language preprocessing:
# 1. We lowered and tokenized our sentences. (we used a RegEx Tokenizer)
# 2. We checked if a word is an abbreviation and replaced it with its full description for consistency among all documents (get_abbreviation_lexicon()).
# 3. We checked if a word is a "stopword" and erased it from the corpus. (we used the NLTK corpus)
# 4. We stemmed our words. (we used the PorterStemmer)
# 
# After preprocessing we broke down the documents in four buckets, since one big bucket would max out the given RAM and saved it as a pickle file (gave us the possibility to save data as binary). 

# In[ ]:


def save_prep_data():
    """
    preprocesses all relevant json file papers and stores them in a pandas data frame
    """
    all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

    df_covid = pd.DataFrame(columns=['paper_id', 'title', 'abstract', 'body_text'])
    
    batch_size = 20000
    batch_idx = 0
    for idx, entry in enumerate(all_json):
        print(f'Processing index: {idx} of {len(all_json)}')
        
        with open(entry) as json_file:
            json_text = json.load(json_file)

            paper_id = json_text['paper_id']

            body_text_stem = []
            abb_lexicon = {}
            for body_elem in json_text['body_text']:
                abb_lexicon.update(get_abbreviation_lexicon(body_elem['text']))

                for w in tokenizer.tokenize(body_elem['text'].lower()):
                    if w in abb_lexicon:
                        abb = abb_lexicon[w]
                        for abb_w in tokenizer.tokenize(abb):
                            if abb_w not in stop_words:
                                body_text_stem.append(stemmer.stem(abb_w)) 
                    else:
                        if w not in stop_words:
                            body_text_stem.append(stemmer.stem(w)) 

            df_covid.loc[idx] = [paper_id, title, abstract, body_text_stem]
            
        
        if idx % batch_size == 0 and idx != 0:
            print('Save batch #', batch_idx)
            df_covid.to_pickle('prep_data_'+str(batch_idx)+'.pkl')
            df_covid = pd.DataFrame(columns=['paper_id', 'title', 'abstract', 'body_text'])
            batch_idx += 1

    df_covid.to_pickle('prep_data_'+str(batch_idx)+'.pkl')

    
def get_abbreviation_lexicon(text):
    """
    text: full text to create the abbreviation lexicon from
    
    returns a dictionary [k: abbreviation (string); v: specification(list<string>)]
    """
    sentences = sent_tokenize(text)
    abbreviation_lexicon = {}
    
    for sent in sentences:
        bracket_strings = re.findall(r'\((.*?)\)', sent)
        
        p_symbol = re.compile('[-,@_!#$%^&*()<>?/\|}{~:]|\d|[\u0370-\u03FF]') # Check for symbols and numbers
        p_uppercase = re.compile('\w+[A-Z]') # Only use uppercase letters
        
        if bracket_strings:
            for string in bracket_strings:
                if len(string) > 1 and len(string.split()) == 1 and re.search(p_symbol, string) == None:
                    uc_string = re.search(p_uppercase, string)
                    if uc_string:
                        string = uc_string.group(0)
                        
                        re_spec = None
                        pattern = '\((?='+ string + ')'
                        i = 0
                        while True:
                            letter = string[len(string)-(i+1)]
                            
                            if i == 0: 
                                pattern = '\w+ ' + pattern 
                            else: 
                                pattern = '\w+( |-)' + pattern   
                            
                            re_result = re.search(pattern, sent)
                            if not re_result:
                                pattern = '\((?='+ string + ')'
                                for i in range(len(string)):
                                    if i == 0: 
                                        pattern = '\w+ ' + pattern 
                                    else: 
                                        pattern = '\w+( |-)' + pattern  
                                    re_spec = re.search(pattern, sent)   
                                break
                                
                            first_letter = re_result.group(0)[0]
                            
                            if first_letter.lower() == letter.lower():
                                i += 1
                                if i == len(string):
                                    re_spec = re.search(pattern, sent)  
                                    break
                        if re_spec:
                            specification = re_spec.group(0)[:-2]
                            abbreviation_lexicon.update({string.lower(): specification.lower()})
    return abbreviation_lexicon


# # TF-IDF Calculation:
# 
# In order to find useful documents based on our word vector we leverage the tf-idf formular. Thereby, we considered one paper as a bag of words. The TF-IDF Formular it self is sensitive to occurencies of specific words within this bag. By leveraging an abbreviation lexicon we tried to compensate the difference between the information input from the user and the term specific knowledge of scientists. 
# 
# Within the formular we decided to go with the log-normalized TF value and the smooth inverse document frequency. By using this slight modification of the formular we were able to standardize our metric among the highly fluctuative length of documents. 
# 
# In a mathmatical way the formular looks as following: 
# <img src="https://dxlcdg.db.files.1drv.com/y4mGJgWxDwCygv1uY1RcE8DXA_p6eI4FwAcKywLIYNVj_u7YvDeIWebCJpDLw8wNfLAZfbY3D9i5fWNX_VXrE0DxOJZmN3Kz_7ifGhyXW7P04_qmfENVv3WxYmdJ-3Gs5Msq6wNc2RqlA8dv5mPENk9fWW7xpwO8WDHAJyaIRv9P9mGOnbCs9xFV6FUDJneSM1YTmp3jElPgbHaeUFh8-dGlw?width=1992&height=1125&cropmode=none" width="1992" height="1125" />

# In[ ]:


def tf_idf(x, D):
    """
    x: word to calculate the tf-idf value (string)
    D: corpus with all documents (list<list<string>>)
    
    returns the tfidf-value for x in D (double)
    """
    x = PorterStemmer().stem(x)
    
    # calculate the idf value
    texts = D['body_text']
    N = len(texts)
    d_count = len(texts[texts.str.contains(x)])
    idf_val = math.log10(N / (1 + d_count)) + 1
    
    result = []
    
    for _, row in D.iterrows():
        if len(row['body_text']) > 0:
            # calculate the tf value
            tf_val = math.log10(1 + (row['body_text'].count(x) / len(row['body_text'])))
            # combine tf- and idf-value
            tf_idf = tf_val * idf_val
            result.append({'paper_id': row['paper_id'], 'tf-idf': tf_idf})
            
    return result

def sum_tf_idf(X, D, topN, sum_results):
    """
    X: Wordlist to calculate the tf-idf values (list<string>)
    D: corpus with all documents (list<list<string>>)
    topN: number of the top N documents to return based on the tfidf 
    
    returns a result dictionary. [paper_id (string), tf-idf (double)]
    """
    tfidf_results = []
    
    
    for x in X:
        tfidf = tf_idf(x, D)
        tfidf = sorted(tfidf, key=lambda k: k['paper_id'], reverse=True) 
        tfidf_results.append(tfidf)

    for i in range(len(tfidf_results[0])):
        n = 0
        value = 0
        for res in tfidf_results:
            if res[i]['tf-idf'] > 0:
                n += 1
            value += res[i]['tf-idf']
        value *= n

        if len(sum_results) < topN:
            sum_results.append({'paper_id': tfidf_results[0][i]['paper_id'], 'tf-idf': value})
            sum_results = sorted(sum_results, key=lambda k: k['tf-idf'], reverse=True) 
        elif value > sum_results[topN-1]['tf-idf']:
            sum_results[topN-1] = {'paper_id': tfidf_results[0][i]['paper_id'], 'tf-idf': value}
            sum_results = sorted(sum_results, key=lambda k: k['tf-idf'], reverse=True) 
    return sum_results


# # Functions to access N-most relevant documents based on their TF-IDF value
# 
# Here we created a simply function that returns the n-most relevant documents for the given search vector, based on their TF-IDF values. 

# In[ ]:


def get_most_relevant(X, topN=20):
    """
    X: Wordlist to calculate the sum-tf-idf values (list<string>)
    topN: number of the top N documents to return based on the tfidf (integer)
    
    returns a result dictionary for the N-most relevant papers. [paper_id (string), tf-idf (double)]
    """
    sum_results = []
    
    for i in range(3):
        data = pd.read_pickle('../input/preprocessed-data/prep_data/prep_data_' + str(i) + '.pkl')
        data = data[data['paper_id'].str[0:3] != 'PMC']
        sum_results = sum_tf_idf(X, data, topN, sum_results)
        del data
        gc.collect()
    return sum_results


# # Functions to access specific information within data files
# 
# For verification and further investigation of the relevant papers we created a few simple functions that let us acces various data points in the documents. For example we get_abstract() gives you the abstract of the paper, if there exists any.

# In[ ]:


def get_body_text(paper_id):
    """
    paper_id: ID of the paper (string)
    
    returns the body_text (string)
    """
    file = glob.glob('/kaggle/input/CORD-19-research-challenge/**/' + paper_id + '.json', recursive=True)
    
    body_text = ''
    if len(file) > 0:
        file = json.load(open(file[0]))
        
        for body_elem in file['body_text']:
            body_text += body_elem['text']
    
    return body_text

def get_abstract(paper_id):
    """
    paper_id: ID of the paper (string)
    
    returns the abstract (string)
    """
    file = glob.glob('/kaggle/input/CORD-19-research-challenge/**/' + paper_id + '.json', recursive=True)
    
    abstract = ''
    if len(file) > 0:
        file = json.load(open(file[0]))
        
        if 'abstract' in file:
            for body_elem in file['abstract']:
                abstract += body_elem['text'] 
    return abstract

def get_title(paper_id):
    """
    paper_id: ID of the paper (string)
    
    returns the title (string)
    """
    file = glob.glob(f'{root_path}/**/' + paper_id + '.json', recursive=True)
    
    title = '**No title found**'
    if len(file) > 0:
        file = json.load(open(file[0]))
        title = file['metadata']['title']
    return title


# # Functions to generate summaries and detect similar sentences
# 
# In order to give the reader a short executive summary for the investigated paper we created three different summary functions. 
# 
# Thereby we tried to satisfy three different focus points for summaries: 
# 
# 1. General overview of the text                                        (satisfied by the gensim summary function)
# 2. Keywords specific insight to narrow focuspoints within the document (satisfied by our keyword focused summary)
# 3. A good scientific overview of the paper                             (satisfied by the SIF-embedding summary)

# In[ ]:


from heapq import nlargest
from gensim.summarization.summarizer import summarize


def get_gensim_summary(text, ratio=0.2):
    """
    text: The original text to be summarized (string)
    ratio: defines how huge the summary should be based on percentage of the original text (double)
    
    returns a text summary based on the gensim summary function (string)
    """
    return summarize(text, ratio=ratio)

def get_sif_summary(text, model, compare_text=None, num_sentences=10, as_array=False):
    """
    text: the text to summarize (string)
    model: gensim model to get the word vectors (BaseWordEmbeddingsModel e.g. word2vec, fasttext) 
    compare_text: text to compute the similarity of the sentences (e.g. paper's abstract)
    num_sentences: length of the summary (int)
    as_array: True -> return summary as array; False: return as text
    
    returns the summary (list<string> or string)
    """
    if text == '' or text == None:
        return 'No text found. Summarization failed!'
    
    sentences = sent_tokenize(text)  #split text into sentences

    if compare_text == None:
        compare_text = text
    text_sif = sif_embeddings([compare_text], model)  #Calculate sif-vector for the text

    similarity = []

    for sent in sentences:
        sent_sif = sif_embeddings([sent], model)  #Calculate sif-vector for the sentence

        sif = np.concatenate((text_sif, sent_sif), axis=0)
        cos = cosine_similarity(sif)[0][1]
        
        similarity.append({'sent': sent, 'cos': cos})

    similarity = sorted(similarity, key=lambda k: k['cos'], reverse=True)

    summary = ''
    if as_array:
        summary = []
    for i in range(num_sentences):
        if as_array:
            summary.append(similarity[i]['sent'])
        else:
            summary += similarity[i]['sent'] + '\n'
    return summary

def get_sentence_similarity(text1, text2, model):
    """
    text1, text2: texts to calculate the cosine similarity, list of sentences in text (string)
    model: gensim model to get the word vectors (BaseWordEmbeddingsModel e.g. word2vec, fasttext) 
    
    returns a result dictionary. [sent1 (string), sent2 (string), cos (double)]
    """
    sent_similarity = []
    
    sentences1 = sent_tokenize(text1)
    sentences2 = sent_tokenize(text2)
    
    manager = Manager()
    emb_dict1 = manager.dict()
    emb_dict2 = manager.dict()
    
    def calc_sif1(sentences):
        for s in sentences:
            emb_dict1[s] = sif_embeddings([s], model)
    def calc_sif2(sentences):
        for s in sentences:
            emb_dict2[s] = sif_embeddings([s], model)    
    p1 = Process(target=calc_sif1, args=(sentences1,))
    p2 = Process(target=calc_sif2, args=(sentences2,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    
    for k1, v1 in emb_dict1.items():
        for k2, v2 in emb_dict2.items():
            sif = np.concatenate((v1, v2))
            cos = cosine_similarity(sif)[0][1]
            sent_similarity.append({'sent1': k1, 'sent2': k2, 'cos': cos})
    sent_similarity = sorted(sent_similarity, key=lambda k: k['cos'], reverse=True) 
    return sent_similarity

def sif_embeddings(texts, model, alpha=1e-3):
    """
    Calculates sentence embeddings based on the avg sif-word-vectors in the sentence
    
    texts: list of texts to perform sentence embedding (list<string>)
    model: gensim model to get the word vectors (BaseWordEmbeddingsModel e.g. word2vec, fasttext)
    alpha: alpha value for the sif calculation, standard=0,003 (double)
    
    returns a vector for each sentence (numpy.ndarray)
    """
    tok_texts = []
    for i in range(len(texts)):
        tok_texts.append(tokenizer.tokenize(texts[i]))
    
    vlookup = model.wv.vocab
    vectors = model.wv 
    size = model.vector_size

    # Compute the normalization constant Z
    #Z = sum(len(t) for t in tok_texts)
    Z = sum(vlookup[k].count for k in vlookup)
    
    output = []
    
    for t in tok_texts:
        count = 0
        v = np.zeros(size, dtype=np.float32)  # Summary vector
        
        for w in t:
            if w in vlookup:
                v += (alpha / (alpha + (vlookup[w].count / Z))) * vectors[w]
                count += 1

        if count > 0:
            for i in range(size):
                v[i] *= 1 / count
        output.append(v)
    
    return np.vstack(output).astype(np.float32)


# # Step 1 : Get the most relevant Papers based on a given search vector

# In[ ]:


results = get_most_relevant(['social', 'distancing'], topN=10)
corpus = []
for i in range(len(results)):
    text = get_body_text(results[i]['paper_id'])
    corpus.append(text)


# # Step 2 : Show similar papers for the given search vector
# 
# This steps helps you to identify similar papers using a heatmap visualization. 
# 
# For identifying papers that are similar we use cosine similarity, a direction measure for vectors. For this the text has to be embedded as a vector. The words are embedded using the word2vec model by gensim. For identifying the similarity between whole papers we combine the word-embeddings with the Smooth Inverse Frequency(SIF) approach, by averaging the sum of all word vectors of a text. 
# 
# Word embeeding allows us to span dimensional feature spaces for the documents and thus we are able to use linear algebra techniques from machine learning to measure the distance of documents within these dimensional spaces.
# 
# Papers that cover the same topics are likely to have a high cosine similarity, while documents from diffent topics have a low correlation or even negative.
# 
# For comprehensive visual explanation:
# <img src="https://ehlkdg.db.files.1drv.com/y4mIcBzGGmA-S_Cq2YQ0djO-LiIliShqTK9bRhgdShsrg1GPYcVuinTlwQC7KZZ253Qbhp1nvGrzTkhXRx-cnAL6D2aw6NByNyYfaIm1BSgZP_P7f6Z5ep5hid4BN4ZxSj7uZsD8p3BjP6JuNdJHn2QGvgABcgMYTwTRKIDeM7rM-raAV1bRqLawkJTMtzyeSynXLEiMqyd8_ypycbGIBEF_Q?width=2538&height=1305&cropmode=none" width="2538" height="1305" />

# In[ ]:


w2v_sif = sif_embeddings(corpus, w2v)
matrix = cosine_similarity(w2v_sif)

import seaborn as sns
import matplotlib.pyplot as plt

mask = np.zeros_like(matrix)
mask[np.triu_indices_from(mask)] = True

sns.set()
sns.set_style('white')
a4_dims = (20, 12)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.heatmap(ax=ax, data=matrix, mask=mask, annot=True, fmt=".2f")

titles = []
for result in results:
    titles.append(get_title(result['paper_id']))
print('Paper titles:')
for idx in range(len(titles)):
    print(str(idx) + ': ' + titles[idx])


# # Step 3 : Generate summaries for the papers you are interested in
# 
# Here you can generate your summary for the text you are interested in. If you use our normal summary function you can indicate specific keywords in which you are interested in. For example you could specifically skim through the text searching for "effecive measurement techniques" in papers about "social distancing". 

# Here you can select which paper you want to summarize. Select the id from the heatmap and title-list above.

# In[ ]:


paper_id = 3


# In[ ]:


# Step 3
abstract = get_abstract(results[paper_id]['paper_id'])
if abstract == '':
    abstract = None
    
sif_summary = get_sif_summary(corpus[paper_id], w2v, compare_text=abstract, num_sentences=5, as_array=False)
gensim_summary = get_gensim_summary(corpus[paper_id], ratio=0.05)


# In[ ]:


print(sif_summary)


# In[ ]:


print(gensim_summary)


# # Step 4 : Monitor if papers have similar opinions on the investigated topic
# 
# What is actually important to know is how many scholar came to the same findings or support certain findings. Thus one can see whether a statements tends to be true or false. 
# 
# In order to help users to easily find those overlaps in documents we build a sentence similarity function that highlights similar sentences within documents. 
# 
# Please select two papers you want to compare.

# In[ ]:


paper_id1 = 2
paper_id2 = 7


# In[ ]:


# Step 4
sent_similarity = get_sentence_similarity(corpus[paper_id1], corpus[paper_id2], w2v)

topN = 5
for item in sent_similarity:
    topN -= 1
    print(str(item['cos']) + ':')
    print()
    print(item['sent1'])
    print()
    print(item['sent2'])
    print()
    print()
    if topN == 0:
        break


# Hurray! You made it to the end. We hope that you could find some interesting ideas in this kernel and that our tool can help you to filter interesting insight from papers and to foster rapid research sharing among scholars. 
